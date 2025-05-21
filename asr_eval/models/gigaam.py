from dataclasses import dataclass

import typing
from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]
from gigaam.decoding import CTCGreedyDecoding # pyright: ignore[reportMissingTypeStubs]
import torch
import numpy as np
import numpy.typing as npt

@dataclass
class GigaamCTCOutputs:
    encoded: torch.Tensor
    encoded_len: torch.Tensor
    log_probs: torch.Tensor
    labels: torch.Tensor
    tokens: list[int]
    text: str

def transcribe_with_gigaam_ctc(
    model: GigaAMASR,
    waveform: npt.NDArray[np.float64],
) -> GigaamCTCOutputs:
    '''
    Pass through Gigaam encoder, gigaam head, then decode and return all the results.
    Sampling rate should be equal to gigaam.preprocess.SAMPLE_RATE == 16_000.
    '''
    assert isinstance(model.decoding, CTCGreedyDecoding)
    
    waveform_tensor = torch.tensor(waveform, dtype=model._dtype).unsqueeze(0) # pyright: ignore[reportPrivateUsage]
    length = torch.tensor([waveform_tensor.shape[1]])
    
    encoded, encoded_len = model.forward(waveform_tensor, length)
    
    log_probs = typing.cast(torch.Tensor, model.head(encoder_output=encoded))
    labels: log_probs.argmax(dim=-1, keepdim=False)
    
    skip_mask = labels != model.decoding.blank_id
    skip_mask[:, 1:] = torch.logical_and(skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1])
    skip_mask[encoded_len:] = 0
    
    tokens: list[int] = labels[0][skip_mask[0]].cpu().tolist() # pyright: ignore[reportUnknownMemberType]
    text = "".join(model.decoding.tokenizer.decode(tokens))
    
    return GigaamCTCOutputs(
        encoded=encoded,
        encoded_len=encoded_len,
        log_probs=log_probs,
        labels=labels,
        tokens=tokens,
        text=text,
    )