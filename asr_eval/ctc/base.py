from __future__ import annotations

from typing import overload
from itertools import groupby

@overload
def ctc_mapping(symbols: list[str], blank: str) -> list[str]: ...

@overload
def ctc_mapping(symbols: list[int], blank: int) -> list[int]: ...

def ctc_mapping(symbols, blank):
    '''
    Represent a CTC mapping. First removes duplicates, then removes blank tokens.
    
    ```
    x = list('_________дджжой   иссто__ч_ни__ки_________   _иссто_ри__и')
    assert ctc_mapping(x, blank='_') == list('джой источники истории')
    ```
    '''
    return [key for key, _group in groupby(symbols) if key != blank]

# def visualize(
#     waveform: npt.NDArray[np.floating],
#     symbols: list[str | int],
#     n_seconds: float | None = None,
#     figsize: tuple[float, float] = (15, 2),
#     sampling_rate: int = 16_000,
#     tokens_freq: float = 25.0,
# ):
#     if n_seconds is None:
#         n_seconds = len(waveform) / sampling_rate
#     else:
#         n_seconds = min(n_seconds, len(waveform) / sampling_rate)
        
#     waveform = waveform[:int(sampling_rate * n_seconds)]
#     ticks = np.arange(0, sampling_rate * n_seconds, int(sampling_rate / tokens_freq))
    
#     plt.figure(figsize=figsize) # type: ignore
#     plt.plot(waveform) # type: ignore
#     plt.xlim(0, n_seconds * sampling_rate) # type: ignore
#     plt.gca().set_xticks(ticks) # type: ignore
#     plt.gca().set_xticklabels(symbols) # type: ignore
#     plt.yticks([]) # type: ignore
#     plt.show() # type: ignore
    
#     IPython.display.display(IPython.display.Audio(waveform, rate=sampling_rate)) # type: ignore