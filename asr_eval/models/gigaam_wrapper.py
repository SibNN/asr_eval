from dataclasses import dataclass
from typing import Any, cast, override
import warnings

import gigaam
from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD
from gigaam.decoding import CTCGreedyDecoding
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from ..ctc.base import ctc_mapping
from ..ctc.chunking import average_logp_windows, chunked_ctc_prediction
from .base import ASREvalWrapper
from ..utils.types import FLOATS, INTS


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
        tokens: list[int] = labels[i][skip_mask[i]].cpu().tolist() # pyright: ignore[reportUnknownMemberType]
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


class GigaAMWrapper(ASREvalWrapper):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.model: GigaAMASR | None = None
        
    def _forward(self, waveform: FLOATS) -> FLOATS:
        assert self.model is not None
        return transcribe_with_gigaam_ctc(self.model, [waveform])[0].log_probs
    
    @override
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        self.model = self.model or cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
        texts: list[str] = []
        for waveform in waveforms:
            predicted_windows = chunked_ctc_prediction(
                waveform=waveform,
                ctc_model=self._forward,
                model_tick_size_sec=1 / FREQ,
                segment_size_sec=30,
                segment_shift_sec=6,
                sampling_rate=SAMPLE_RATE,
            )
            merged_log_probs = average_logp_windows(predicted_windows)
            labels = cast(list[int], merged_log_probs.argmax(axis=-1, keepdims=False).tolist())
            tokens = ctc_mapping(labels, self.model.decoding.blank_id)
            texts.append(decode(self.model, tokens))
            
        return texts