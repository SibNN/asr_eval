from dataclasses import dataclass

import typing
import warnings
from gigaam.model import GigaAMASR, LONGFORM_THRESHOLD # pyright: ignore[reportMissingTypeStubs]
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
    
    if len(waveform) / 16_000 > LONGFORM_THRESHOLD:
        warnings.warn("too long audio, GigaAMASR.transcribe() would throw an error", RuntimeWarning)
    
    waveform_tensor = torch.tensor(waveform, dtype=model._dtype).unsqueeze(0) # pyright: ignore[reportPrivateUsage]
    length = torch.tensor([waveform_tensor.shape[1]])
    
    encoded, encoded_len = model.forward(waveform_tensor, length)
    
    log_probs = typing.cast(torch.Tensor, model.head(encoder_output=encoded))
    labels = log_probs.argmax(dim=-1, keepdim=False)
    
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

def decode_each_token(model: GigaAMASR, tokens: list[int] | torch.Tensor) -> list[str]:
    '''
    Example:
    model = gigaam.load_model('ctc', device='cpu')
    outputs = transcribe_with_gigaam_ctc(model, waveform)
    symbols = decode_each_token(model, outputs.labels[0])
    print(''.join(symbols))
    >>> '___и по_эттому  иисполльзо_ватьь иих вв по_ввседдне .....
    text = ''.join([key for key, _group in groupby(symbols) if key != '_'])
    print(text)
    >>> 'и поэтому использовать их в повседневности .....
    assert text == outputs.text
    >>> True
    '''
    if isinstance(tokens, torch.Tensor):
        assert tokens.ndim == 1, 'pass a single sample, not a batch'
        tokens = tokens.cpu().tolist() # type: ignore
    return [
        model.decoding.tokenizer.decode([x])
        if x != model.decoding.blank_id
        else '_'
        for x in tokens
    ]