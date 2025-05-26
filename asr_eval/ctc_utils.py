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
    def _forced_alignment(
        log_probs_idx: int = 0,
        tokens_idx: int = 0,
        _prev_token_id: int = blank_id
    ) -> tuple[list[int], list[float]]:
        if log_probs_idx >= len(log_probs):
            return [], [0 if tokens_idx >= len(tokens) else -np.inf]
        
        # we will consired several paths and then select one with the highest probability
        # each path is a list of token ids (possibly including blank_id) and their log probabilities
        paths: list[tuple[list[int], list[float]]] = []
        
        # option 1: blank token is selected for the first frame
        tail_tokens, tail_p = _forced_alignment(log_probs_idx + 1, tokens_idx, blank_id)
        paths.append((
            [blank_id] + tail_tokens,
            [log_probs[log_probs_idx, blank_id]] + tail_p,
        ))
        
        # option 2: prev token is selected for the first frame (!= blank token)
        if _prev_token_id != blank_id:
            tail_tokens, tail_p = _forced_alignment(log_probs_idx + 1, tokens_idx, _prev_token_id)
            paths.append((
                [_prev_token_id] + tail_tokens,
                [log_probs[log_probs_idx, _prev_token_id]] + tail_p,
            ))
        
        # option 3: true_tokens[0] is selected for the first frame (!= prev token)
        if tokens_idx < len(tokens) and tokens[tokens_idx] != _prev_token_id:
            tail_tokens, tail_p = _forced_alignment(log_probs_idx + 1, tokens_idx + 1, tokens[tokens_idx])
            paths.append((
                [tokens[tokens_idx]] + tail_tokens,
                [log_probs[log_probs_idx, tokens[tokens_idx]]] + tail_p,
            ))
        
        return max(paths, key=lambda path: sum(path[1]))
    
    result_tokens, result_p = _forced_alignment(0, 0, blank_id)
    if sum(result_p) == -np.inf:
        raise RuntimeError('cannot perform a force algnment')
    return result_tokens, result_p[:-1]  # drop the latest 0