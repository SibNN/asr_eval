from dataclasses import dataclass
from typing import Literal, cast, override
import warnings

import gigaam
from gigaam.vad_utils import segment_audio as gigaam_segment_audio
from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD
from gigaam.decoding import CTCGreedyDecoding
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Literal
from ..ctc.base import ctc_mapping
from ..segments.chunking import chunk_audio, average_segment_features
from .base import Transcriber
from ..utils.types import FLOATS, INTS

SAMPLING_RATE = 16000
FREQ = 25  # GigaAM2 encoder outputs per second

@dataclass
class GigaamCTCOutputs:
    encoded: FLOATS
    log_probs: FLOATS  # exp(log_probs) sums to 1
    text: str
    
@torch.inference_mode()
def transcribe_with_gigaam_ctc(
    model: GigaAMASR,
    waveforms: list[FLOATS],
) -> list[GigaamCTCOutputs]:
    '''
    Pass through Gigaam encoder, gigaam head, then decode and return all the results.
    Sampling rate should be equal to gigaam.preprocess.SAMPLE_RATE == 16_000.
    '''
    assert isinstance(model.decoding, CTCGreedyDecoding)
    
    for waveform in waveforms:
        if len(waveform) / SAMPLE_RATE > LONGFORM_THRESHOLD:
            warnings.warn("too long audio, GigaAMASR.transcribe() would throw an error", RuntimeWarning)
    
    waveform_tensors = [
        torch.tensor(w, dtype=model._dtype).to(model._device) # pyright: ignore[reportPrivateUsage]
        for w in waveforms
    ]
    lengths = torch.tensor([len(w) for w in waveforms]).to(model._device) # pyright: ignore[reportPrivateUsage]
    
    waveform_tensors_padded = pad_sequence(
        waveform_tensors,
        batch_first=True,
        padding_value=0,
    )
    
    encoded, encoded_len = model.forward(waveform_tensors_padded, lengths)
    
    # exp(log_probs) sums to 1
    # GigaamCTCOutputs will check this on __post_init__
    log_probs = cast(torch.Tensor, model.head(encoder_output=encoded))
    labels = log_probs.argmax(dim=-1, keepdim=False)
    
    skip_mask = labels != model.decoding.blank_id
    skip_mask[:, 1:] = torch.logical_and(skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1])
    
    for i in range(len(labels)):
        skip_mask[i, encoded_len[i]:] = 0
    
    results: list[GigaamCTCOutputs] = []
    
    for i in range(len(labels)):
        tokens: list[int] = labels[i][skip_mask[i]].cpu().tolist() # type: ignore
        text = "".join(model.decoding.tokenizer.decode(tokens))
        
        results.append(GigaamCTCOutputs(
            encoded=encoded[i][:encoded_len[i], :].cpu().numpy(), # type: ignore
            log_probs=log_probs[i][:encoded_len[i], :].cpu().numpy(), # type: ignore
            text=text,
        ))
    
    return results

def decode(model: GigaAMASR, tokens: list[int] | INTS) -> str:
    if isinstance(tokens, np.ndarray):
        assert tokens.ndim == 1, 'pass a single sample, not a batch'
        tokens = tokens.tolist()
    return ''.join([
        model.decoding.tokenizer.decode([x])
        if x != model.decoding.blank_id
        else '_'
        for x in tokens
    ])


class GigaAMEncodeError(ValueError):
    pass


def encode(model: GigaAMASR, text: str) -> list[int]:
    assert model.decoding.tokenizer.charwise
    tokens: list[int] = []
    for char in text:
        if not char in model.decoding.tokenizer.vocab:
            raise GigaAMEncodeError(f'Cannot encode char "{char}": does not exist in vocab')
        tokens.append(model.decoding.tokenizer.vocab.index(char))
    return tokens


class GigaAMWrapper(Transcriber):
    longform_mode: Literal['vad', 'uniform']

    def __init__(
        self,
        longform_mode: Literal['vad', 'uniform'] = 'vad',
    ):
        self.model: GigaAMASR | None = None
        self.longform_mode = longform_mode
        
    def _single_forward(self, waveform: FLOATS) -> str:
        assert self.model is not None
        wav = (
            torch.tensor(waveform)
            .to(self.model._device)  # pyright:ignore[reportPrivateUsage]
            .to(self.model._dtype)  # pyright:ignore[reportPrivateUsage]
            .unsqueeze(0)
        )
        length = torch.full([1], wav.shape[-1], device=self.model._device)  # pyright:ignore[reportPrivateUsage]
        encoded, encoded_len = self.model.forward(wav, length)
        return self.model.decoding.decode(self.model.head, encoded, encoded_len)[0]
    
    def _single_forward_ctc(self, waveform: FLOATS) -> FLOATS:
        assert self.model is not None
        return transcribe_with_gigaam_ctc(self.model, [waveform])[0].log_probs

    def transcribe_longform_uniform(self, waveform: FLOATS) -> str:
        assert self.model is not None
        segments = chunk_audio(
            len(waveform) / SAMPLE_RATE,
            segment_length=30,
            segment_shift=10
        )
        log_probs = [self._single_forward_ctc(waveform[segment.slice()]) for segment in segments]
        merged_log_probs = average_segment_features(
            segments=segments,
            features=log_probs,
            feature_tick_size=1 / FREQ,
        )
        
        labels = cast(list[int], merged_log_probs.argmax(axis=-1, keepdims=False).tolist())
        tokens = ctc_mapping(labels, self.model.decoding.blank_id)
        return decode(self.model, tokens)

    def transcribe_longform_vad(self, waveform: FLOATS) -> str:
        assert self.model is not None
        segments_tensors: list[torch.Tensor]
        _boundaries: list[tuple[float, float]]

        maxval = np.abs(waveform).max()
        if maxval > 0:
            waveform /= maxval
        
        torch_int_waveform = torch.tensor(waveform * 32768, dtype=torch.int16).clone()
        segments_tensors, _boundaries = gigaam_segment_audio(
            torch_int_waveform,
            SAMPLE_RATE,
            max_duration=22.,
            min_duration=15.,
            new_chunk_threshold=0.2,
            device=self.model._device,  # pyright:ignore[reportPrivateUsage]
        )
        print(_boundaries)
        print([x.shape for x in segments_tensors])
        transcriptions = [
            self._single_forward(seg.numpy()) # type: ignore
            for seg in segments_tensors
        ]
        return ' '.join(transcriptions)

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        self.model = self.model or cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
        match self.longform_mode:
            case 'vad':
                return self.transcribe_longform_vad(waveform)
            case 'uniform':
                return self.transcribe_longform_uniform(waveform)