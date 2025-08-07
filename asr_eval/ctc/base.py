from __future__ import annotations

from typing import overload
from itertools import groupby


__all__ = [
    'ctc_mapping',
]


@overload
def ctc_mapping(symbols: list[str], blank: str) -> list[str]: ...

@overload
def ctc_mapping(symbols: list[int], blank: int) -> list[int]: ...

def ctc_mapping(symbols, blank): # type: ignore
    '''
    Represent a CTC mapping. First removes duplicates, then removes blank tokens.
    
    ```
    x = list('_________дджжой   иссто__ч_ни__ки_________   _иссто_ри__и')
    assert ctc_mapping(x, blank='_') == list('джой источники истории')
    ```
    '''
    return [key for key, _group in groupby(symbols) if key != blank] # type: ignore