from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torchaudio

from asr_eval.utils.misc import groupby_into_spans


def forced_alignment(
    log_probs: npt.NDArray[np.floating[Any]],
    true_tokens: list[int] | npt.NDArray[np.integer[Any]],
    blank_id: int = 0,
) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    '''
    Returns the path with the highest cumulative probability among all paths
    that match the specified transcription.
    
    A wrapper around the pytorch implementation.
    
    Returns:
    - token for each frame
    - probability for each frame
    - frame span (start_pos, end_pos) for each of true_tokens
    '''
    alignments, scores = torchaudio.functional.forced_align(
        torch.tensor(log_probs).unsqueeze(0),
        torch.tensor(true_tokens, dtype=torch.int32, device='cpu').unsqueeze(0),
        blank=blank_id,
    )
    idx_per_frame: list[int] = alignments[0].tolist() # type: ignore
    scores_per_frame: list[float] = scores[0].tolist() # type: ignore
    
    spans = [
        (value, start, end)
        for value, start, end in groupby_into_spans(idx_per_frame)
        if value != blank_id
    ]
    assert [value for value, _start, _end in spans] == true_tokens
    true_tokens_pos = [(start, end) for _value, start, end in spans]
    
    return idx_per_frame, scores_per_frame, true_tokens_pos


def recursion_forced_alignment(
    log_probs: npt.NDArray[np.floating[Any]],
    tokens: list[int] | npt.NDArray[np.integer[Any]],
    blank_id: int = 0,
) -> tuple[list[int], list[float]]:
    '''
    Returns the path with the highest cumulative probability among all paths
    that match the specified transcription.
    
    A custom minimal implementation via recursion.
    
    Returns:
    - token for each frame
    - probability for each frame
    '''
    assert all(t != blank_id for t in tokens)
    
    @lru_cache(maxsize=None)
    def _forced_alignment(
        log_probs_pos: int = 0,
        tokens_pos: int = 0,
        prev_token: int = blank_id,
    ) -> tuple[list[int], list[float]]:
        # Performs the forced alignment of log_probs[log_probs_pos:] to tokens[tokens_pos:] given that
        # prev_token_id was selected for the frame log_probs_pos - 1. Will recursively solve via
        # _forced_alignment(log_probs_pos + 1, ...), considering several options for the first frame
        
        if log_probs_pos >= len(log_probs):
            return [], [0 if tokens_pos >= len(tokens) else -np.inf]
        
        # each path is a list of token ids (possibly including blank_id) and their log probabilities
        paths: list[tuple[list[int], list[float]]] = []
        
        # option 1: blank token is selected for the first frame
        tail_tokens, tail_p = _forced_alignment(log_probs_pos + 1, tokens_pos, blank_id)
        paths.append((
            [blank_id] + tail_tokens,
            [log_probs[log_probs_pos, blank_id]] + tail_p,
        ))
        
        # option 2: prev token is selected for the first frame (!= blank token)
        if prev_token != blank_id:
            tail_tokens, tail_p = _forced_alignment(log_probs_pos + 1, tokens_pos, prev_token)
            paths.append((
                [prev_token] + tail_tokens,
                [log_probs[log_probs_pos, prev_token]] + tail_p,
            ))
        
        # option 3: true_tokens[0] is selected for the first frame (!= prev token)
        if tokens_pos < len(tokens) and tokens[tokens_pos] != prev_token:
            tail_tokens, tail_p = _forced_alignment(log_probs_pos + 1, tokens_pos + 1, tokens[tokens_pos])
            paths.append((
                [tokens[tokens_pos]] + tail_tokens,
                [log_probs[log_probs_pos, tokens[tokens_pos]]] + tail_p,
            ))
        
        return max(paths, key=lambda path: sum(path[1]))
    
    result_tokens, result_p = _forced_alignment(0, 0, blank_id)
    if sum(result_p) == -np.inf:
        raise RuntimeError('cannot perform a force algnment')
    return result_tokens, result_p[:-1]  # drop the latest 0