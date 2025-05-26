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
) -> tuple[list[int], list[float]]:
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
    return alignments[0].tolist(), scores[0].tolist(),  # type: ignore

@dataclass
class _FA_Path:
    token_ids: list[int]
    log_p: list[float]

def recursion_ctc_forced_alignment(
    log_probs: npt.NDArray[np.floating],
    tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> tuple[list[int], list[float]]:
    '''
    Returns the path with the highest cumulative probability among all paths
    that match the specified transcription.
    
    A custom minimal implementation via recursion.
    '''
    assert all(t != blank_id for t in tokens)
    
    @lru_cache(maxsize=None)
    def fa(
        log_probs_idx: int = 0,
        tokens_idx: int = 0,
        _prev_token_id: int = blank_id
    ) -> tuple[list[int], list[float]]:
        if log_probs_idx >= len(log_probs):
            return [], [0 if tokens_idx >= len(tokens) else -np.inf]
        
        # option 1: blank token
        tail_tokens, tail_p = fa(log_probs_idx + 1, tokens_idx, blank_id)
        path_1 = _FA_Path(
            [blank_id] + tail_tokens,
            [log_probs[log_probs_idx, blank_id]] + tail_p,
        )
        best_path = path_1
        
        # option 2: prev token that is not blank token
        if _prev_token_id != blank_id:
            tail_tokens, tail_p = fa(log_probs_idx + 1, tokens_idx, _prev_token_id)
            path_2 = _FA_Path(
                [_prev_token_id] + tail_tokens,
                [log_probs[log_probs_idx, _prev_token_id]] + tail_p,
            )
            if sum(path_2.log_p) > sum(best_path.log_p):
                best_path = path_2
        
        # option 3: true_tokens[0] that is not the same as prev token
        if tokens_idx < len(tokens) and tokens[tokens_idx] != _prev_token_id:
            tail_tokens, tail_p = fa(log_probs_idx + 1, tokens_idx + 1, tokens[tokens_idx])
            path_3 = _FA_Path(
                [tokens[tokens_idx]] + tail_tokens,
                [log_probs[log_probs_idx, tokens[tokens_idx]]] + tail_p,
            )
            if sum(path_3.log_p) > sum(best_path.log_p):
                best_path = path_3
        
        return best_path.token_ids, best_path.log_p
    
    result_tokens, result_p = fa(0, 0, blank_id)
    if sum(result_p) == -np.inf:
        raise RuntimeError('cannot perform a force algnment')
    return result_tokens, result_p[:-1]  # drop the latest 0