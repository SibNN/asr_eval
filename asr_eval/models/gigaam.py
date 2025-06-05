from dataclasses import dataclass
import typing
import warnings
import re

from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD # pyright: ignore[reportMissingTypeStubs]
from gigaam.decoding import CTCGreedyDecoding # pyright: ignore[reportMissingTypeStubs]
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import numpy.typing as npt

from ..ctc.base import ctc_mapping
from ..ctc.forced_alignment import forced_alignment


FREQ = 25  # GigaAM2 encoder outputs per second

@dataclass
class GigaamCTCOutputs:
    encoded: npt.NDArray[np.float32 | np.float16]
    log_probs: npt.NDArray[np.float32 | np.float16]  # exp(log_probs) sums to 1
    text: str
    
@torch.inference_mode()
def transcribe_with_gigaam_ctc(
    model: GigaAMASR,
    waveforms: list[npt.NDArray[np.floating]],
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
    log_probs = typing.cast(torch.Tensor, model.head(encoder_output=encoded))
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

def decode(model: GigaAMASR, tokens: list[int] | npt.NDArray[np.int64]) -> str:
    if isinstance(tokens, np.ndarray):
        assert tokens.ndim == 1, 'pass a single sample, not a batch'
        tokens = tokens.tolist()
    return ''.join([
        model.decoding.tokenizer.decode([x])
        if x != model.decoding.blank_id
        else '_'
        for x in tokens
    ])

def encode(model: GigaAMASR, text: str) -> list[int]:
    assert model.decoding.tokenizer.charwise
    tokens: list[int] = []
    for char in text:
        if not char in model.decoding.tokenizer.vocab:
            raise ValueError(f'Cannot encode char "{char}": does not exist in vocab')
        tokens.append(model.decoding.tokenizer.vocab.index(char))
    return tokens

def get_word_timings(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    text: str | None = None,
) -> list[tuple[str, float, float]]:
    '''
    Outputs a list of words and their timings in seconds:

    ([('и', 0.12, 0.16),
        ('поэтому', 0.2, 0.56),
        ('использовать', 0.64, 1.28),
        ('их', 1.32, 1.44),
        ('в', 1.48, 1.56),
        ('повседневности', 1.6, 2.36),
    '''
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    if text is None:
        tokens = outputs.log_probs.argmax(axis=1)
    else:
        tokens, _probs = forced_alignment(
            outputs.log_probs,
            encode(model, text),
            blank_id=model.decoding.blank_id
        )
    letter_per_frame = decode(model, tokens)
    word_timings = [
        (
            ''.join(ctc_mapping(list(match.group()), blank='_')),
            match.start() / FREQ,
            match.end() / FREQ,
        )
        for match in re.finditer(r'[а-я]([а-я_]*[а-я])?', letter_per_frame)
    ]
    return word_timings

