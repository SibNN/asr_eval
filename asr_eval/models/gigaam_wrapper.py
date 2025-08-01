from dataclasses import dataclass
from typing import Literal, cast, override
import warnings

import gigaam
from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD
from gigaam.decoding import CTCGreedyDecoding
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Literal

from .pyannote_segmenter import PyannoteSegmenter
from ..ctc.base import ctc_mapping
from ..segments.chunking import chunk_audio, average_segment_features
from .base import Transcriber
from ..utils.types import FLOATS, INTS

SAMPLING_RATE = 16000
FREQ = 25  # GigaAM2 encoder outputs per second


def transcribe_with_gigaam(model: GigaAMASR, waveform: FLOATS) -> str:
    '''
    Forward for RNNT or CTC model, returns text
    '''
    wav = (
        torch.tensor(waveform)
        .to(model._device)  # pyright:ignore[reportPrivateUsage]
        .to(model._dtype)  # pyright:ignore[reportPrivateUsage]
        .unsqueeze(0)
    )
    length = torch.full([1], wav.shape[-1], device=model._device)  # pyright:ignore[reportPrivateUsage]
    encoded, encoded_len = model.forward(wav, length)
    return model.decoding.decode(model.head, encoded, encoded_len)[0]

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
    Pass through Gigaam encoder, gigaam CTC head, then decode and return all the results.
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


class GigaAMLongformVAD(Transcriber):
    # TODO make more general class
    def __init__(self, head_type: Literal['ctc', 'rnnt'] = 'ctc'):
        self.model = cast(GigaAMASR, gigaam.load_model(head_type, device='cuda'))
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        transcriptions = [
            transcribe_with_gigaam(model, waveform[seg.slice()]) # type: ignore
            for seg in PyannoteSegmenter()(waveform)
        ]
        return ' '.join(transcriptions)
    

class GigaAMLongformUniform(Transcriber):
    # TODO make more general class
    def __init__(self):
        self.model = cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        segments = chunk_audio(
            len(waveform) / SAMPLE_RATE,
            segment_length=30,
            segment_shift=10
        )
        log_probs = [
            transcribe_with_gigaam_ctc(self.model, [waveform[segment.slice()]])[0].log_probs
            for segment in segments
        ]
        merged_log_probs = average_segment_features(
            segments=segments,
            features=log_probs,
            feature_tick_size=1 / FREQ,
        )
        
        labels = cast(list[int], merged_log_probs.argmax(axis=-1, keepdims=False).tolist())
        tokens = ctc_mapping(labels, self.model.decoding.blank_id)
        return decode(self.model, tokens)