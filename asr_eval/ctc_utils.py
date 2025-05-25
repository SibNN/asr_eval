from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

import numpy as np
import numpy.typing as npt
import torch
import torchaudio # pyright: ignore[reportMissingTypeStubs]

def ctc_mapping(symbols: list[str], blank: str = '_') -> list[str]:
    '''
    Represent a CTC mapping. First removes duplicates, then removes blank tokens.
    
    ```
    x = list('_________дджжой   иссто__ч_ни__ки_________   _иссто_ри__и')
    assert ctc_mapping(x) == list('джой источники истории')
    ```
    '''
    return [key for key, _group in groupby(symbols) if key != blank]

def torch_ctc_forced_alignment(
    log_probs: npt.NDArray[np.floating],
    true_tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> tuple[float, list[int]]:
    '''
    Returns the path with the highest cumulative probability among all paths
    that match the specified transcription.
    
    A wrapper around the pytorch implementation.
    '''
    alignments, scores = torchaudio.functional.forced_align(
        torch.tensor(log_probs).unsqueeze(0),
        torch.tensor(true_tokens, dtype=torch.int32, device='cpu').unsqueeze(0),
        blank=blank_id,
    )
    return float(scores.numpy().sum()), alignments[0].tolist() # type: ignore

def recursion_ctc_forced_alignment(
    log_probs: npt.NDArray[np.floating],
    true_tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> tuple[float, list[int]]:
    '''
    Returns the path with the highest cumulative probability among all paths
    that match the specified transcription.
    
    A custom minimal implementation via recursion.
    '''
    @lru_cache(maxsize=None)
    def fa(
        log_probs_start_idx: int = 0,
        true_tokens_start_idx: int = 0,
        _prev_token_id: int | None = None
    ) -> tuple[float, list[int]]:
        
        if _prev_token_id is None:
            _prev_token_id = blank_id
        
        if log_probs_start_idx >= len(log_probs):
            if true_tokens_start_idx >= len(true_tokens):
                return 0, []
            else:
                return -np.inf, []
        
        if true_tokens_start_idx < len(true_tokens):
            assert true_tokens[true_tokens_start_idx] != blank_id
        
        # option 1: blank token
        path_1 = _FA_Path(
            blank_id,
            log_probs[log_probs_start_idx, blank_id],
            *fa(log_probs_start_idx + 1, true_tokens_start_idx, blank_id),
        )
        best_path = path_1
        
        # option 2: prev token that is not blank token
        if _prev_token_id != blank_id:
            path_2 = _FA_Path(
                _prev_token_id,
                log_probs[log_probs_start_idx, _prev_token_id],
                *fa(log_probs_start_idx + 1, true_tokens_start_idx, _prev_token_id),
            )
            if path_2.total_log_p > best_path.total_log_p:
                best_path = path_2
        
        # option 3: true_tokens[0] that is not the same as prev token
        if true_tokens_start_idx < len(true_tokens) and true_tokens[true_tokens_start_idx] != _prev_token_id:
            path_3 = _FA_Path(
                true_tokens[true_tokens_start_idx],
                log_probs[log_probs_start_idx, true_tokens[true_tokens_start_idx]],
                *fa(log_probs_start_idx + 1, true_tokens_start_idx + 1, true_tokens[true_tokens_start_idx]),
            )
            if path_3.total_log_p > best_path.total_log_p:
                best_path = path_3
        
        return best_path.total_log_p, [best_path.current_token_id] + best_path.tail_token_ids
    
    return fa(0, 0, blank_id)

@dataclass
class _FA_Path:
    current_token_id: int
    current_log_p: float
    tail_log_p: float
    tail_token_ids: list[int]
    
    @property
    def total_log_p(self) -> float:
        return self.current_log_p + self.tail_log_p