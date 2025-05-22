from dataclasses import dataclass

import typing
import warnings
from gigaam.model import GigaAMASR, SAMPLE_RATE, LONGFORM_THRESHOLD # pyright: ignore[reportMissingTypeStubs]
from gigaam.decoding import CTCGreedyDecoding # pyright: ignore[reportMissingTypeStubs]
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import numpy.typing as npt
import IPython.display
import matplotlib.pyplot as plt


FREQ = 25  # GigaAM2 encoder outputs per second

@dataclass
class GigaamCTCOutputs:
    encoded: torch.Tensor
    encoded_len: torch.Tensor
    log_probs: torch.Tensor
    labels: torch.Tensor
    tokens: list[int]
    text: str
    decode_each_token: str
    
    def visualize(
        self,
        model: GigaAMASR,
        waveform: npt.NDArray[np.float64],
        n_seconds: float | None = None,
        figsize: tuple[float, float] = (15, 2),
    ):
        symbols1 = decode_each_token(model, self.labels[0])
        symbols2 = decode_each_token(model, self.log_probs[:, :, :-1].argmax(dim=-1)[0])
        
        if n_seconds is None:
            n_seconds = len(waveform) / SAMPLE_RATE
        else:
            n_seconds = min(n_seconds, len(waveform) / SAMPLE_RATE)
            
        waveform = waveform[:int(SAMPLE_RATE * n_seconds)]
        ticks = np.arange(0, SAMPLE_RATE * n_seconds, SAMPLE_RATE // FREQ)
        ticklabels = [
            f'{a}\n{b}' if a == '_' else a
            for a, b in zip(symbols1[:len(ticks)], symbols2[:len(ticks)])
        ]
        
        plt.figure(figsize=figsize) # type: ignore
        plt.plot(waveform) # type: ignore
        plt.xlim(0, n_seconds * SAMPLE_RATE) # type: ignore
        plt.gca().set_xticks(ticks) # type: ignore
        plt.gca().set_xticklabels(ticklabels) # type: ignore
        plt.show() # type: ignore
        
        IPython.display.display(IPython.display.Audio(waveform, rate=SAMPLE_RATE)) # type: ignore

def transcribe_with_gigaam_ctc(
    model: GigaAMASR,
    waveforms: list[npt.NDArray[np.float64]],
) -> list[GigaamCTCOutputs]:
    '''
    Pass through Gigaam encoder, gigaam head, then decode and return all the results.
    Sampling rate should be equal to gigaam.preprocess.SAMPLE_RATE == 16_000.
    '''
    assert isinstance(model.decoding, CTCGreedyDecoding)
    
    for waveform in waveforms:
        if len(waveform) / SAMPLE_RATE > LONGFORM_THRESHOLD:
            warnings.warn("too long audio, GigaAMASR.transcribe() would throw an error", RuntimeWarning)
    
    waveform_tensors = [torch.tensor(w, dtype=model._dtype) for w in waveforms] # pyright: ignore[reportPrivateUsage]
    lengths = torch.tensor([len(w) for w in waveforms])
    
    waveform_tensors_padded = pad_sequence(
        waveform_tensors,
        batch_first=True,
        padding_value=0,
    )
    
    encoded, encoded_len = model.forward(waveform_tensors_padded, lengths)
    
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
            encoded=encoded[i:i+1],
            encoded_len=encoded_len[i:i+1],
            log_probs=log_probs[i:i+1],
            labels=labels[i:i+1],
            tokens=tokens,
            text=text,
            decode_each_token=''.join(decode_each_token(model, labels[i])),
        ))
    
    return results

def decode_each_token(model: GigaAMASR, tokens: list[int] | torch.Tensor) -> list[str]:
    '''
    Example:
    model = gigaam.load_model('ctc', device='cpu')
    outputs, = transcribe_with_gigaam_ctc(model, [waveform])
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