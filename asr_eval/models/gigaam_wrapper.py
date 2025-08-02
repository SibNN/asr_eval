from __future__ import annotations

from typing import cast, override
import warnings

import gigaam
from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD
from gigaam.decoding import CTCGreedyDecoding
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .base import CTC, Transcriber
from ..utils.types import FLOATS, INTS


SAMPLING_RATE = 16000
FREQ = 25  # GigaAM2 encoder outputs per second


class GigaAMShortformBase(Transcriber):
    model: GigaAMASR

    @override
    @torch.inference_mode()
    def transcribe(self, waveform: FLOATS) -> str:
        # a forward + decoding pipeline suitable for both CTC and RNNT
        wav = (
            torch.tensor(waveform)
            .to(self.model._device)  # pyright:ignore[reportPrivateUsage]
            .to(self.model._dtype)  # pyright:ignore[reportPrivateUsage]
            .unsqueeze(0)
        )
        length = torch.full([1], wav.shape[-1], device=self.model._device)  # pyright:ignore[reportPrivateUsage]
        encoded, encoded_len = self.model.forward(wav, length)
        return self.model.decoding.decode(self.model.head, encoded, encoded_len)[0]


class GigaAMShortformRNNT(GigaAMShortformBase):
    def __init__(self):
        self.model = cast(GigaAMASR, gigaam.load_model('rnnt', device='cuda'))


class GigaAMShortformCTC(GigaAMShortformBase, CTC):
    def __init__(self):
        self.model = cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return super(GigaAMShortformBase, self).transcribe(waveform)
    
    @property
    @override
    def blank_id(self) -> int:
        return self.model.decoding.blank_id
    
    @property
    @override
    def tick_size(self) -> float:
        return 1 / FREQ

    @override
    def decode(self, token: int) -> str:
        return self.model.decoding.tokenizer.decode([token])
    
    @override
    @torch.inference_mode()
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        # Sampling rate should be equal to gigaam.preprocess.SAMPLE_RATE == 16_000.
        assert isinstance(self.model.decoding, CTCGreedyDecoding)
        
        for waveform in waveforms:
            if len(waveform) / SAMPLE_RATE > LONGFORM_THRESHOLD:
                warnings.warn("too long audio, GigaAMASR.transcribe() would throw an error", RuntimeWarning)
        
        waveform_tensors = [
            torch.tensor(w, dtype=self.model._dtype).to(self.model._device) # pyright: ignore[reportPrivateUsage]
            for w in waveforms
        ]
        lengths = torch.tensor([len(w) for w in waveforms]).to(self.model._device) # pyright: ignore[reportPrivateUsage]
        
        waveform_tensors_padded = pad_sequence(
            waveform_tensors,
            batch_first=True,
            padding_value=0,
        )
        
        encoded, encoded_len = self.model.forward(waveform_tensors_padded, lengths)
        
        log_probs = cast(torch.Tensor, self.model.head(encoder_output=encoded))
        # exp(log_probs) sums to 1
        
        return [
            _log_probs[:length].cpu().numpy() # type: ignore
            for _log_probs, length in zip(log_probs, encoded_len)
        ]


class GigaAMEncodeError(ValueError):
    pass


def gigaam_encode(model: GigaAMASR, text: str) -> list[int]:
    assert model.decoding.tokenizer.charwise
    tokens: list[int] = []
    for char in text:
        if not char in model.decoding.tokenizer.vocab:
            raise GigaAMEncodeError(f'Cannot encode char "{char}": does not exist in vocab')
        tokens.append(model.decoding.tokenizer.vocab.index(char))
    return tokens


def gigaam_decode(model: GigaAMASR, tokens: list[int] | INTS) -> str:
    if isinstance(tokens, np.ndarray):
        assert tokens.ndim == 1, 'pass a single sample, not a batch'
        tokens = tokens.tolist()
    return ''.join([
        model.decoding.tokenizer.decode([x])
        if x != model.decoding.blank_id
        else '_'
        for x in tokens
    ])