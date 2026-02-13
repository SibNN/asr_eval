from __future__ import annotations
from functools import cache
import os
from typing import cast
import warnings

import numpy as np

from asr_eval.utils.misc import groupby_into_spans
from asr_eval.utils.types import FLOATS, INTS


__all__ = [
    'forced_alignment',
    'forced_alignment_via_recursion',
]


env_var = 'FORCED_ALIGN_CUSTOM'
USE_CUSTOM_IMPLEMENTATION = os.environ.get(env_var, None) == '1'


def forced_alignment(
    log_probs: FLOATS,
    true_tokens: list[int] | INTS,
    blank_id: int = 0,
) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    """Performs a forced alignment.
    
    Returns the path with the highest cumulative probability among all
    paths that match the specified transcription.
    
    Args:
        log_probs: log probabilities from CTC model.
        true_tokens: a sequence of tokens for the ground truth
            transcription.
        blank_id: an index for :code:`<blank>` CTC token.
    
    Returns:
        A tuple. The first element is the token for each frame. The
        second element is a probability for each frame. The third
        element is a frame span (start_position, end_position) for
        each of true_tokens.
    
    Note:
        This is going to stop working for torchaudio>=2.9.0, see
        https://github.com/pytorch/audio/issues/3902 . It is possible
        to use :func:`~asr_eval.ctc.forced_alignment.recursion_forced_alignment`,
        but there may be problems with recursion limit (recursion limit
        is 1000 for Python, equals 20 sec with 50 ticks/sec). To use
        custom implementation, set environmental variable
        :code:`FORCED_ALIGN_CUSTOM=1`.
        
    
    """
    import torch
    import torchaudio
    
    if USE_CUSTOM_IMPLEMENTATION:
        idx_per_frame, scores_per_frame = forced_alignment_via_recursion(
            log_probs=log_probs, true_tokens=true_tokens, blank_id=blank_id,
        )
    elif not hasattr(torchaudio.functional, 'forced_align'):
        raise RuntimeError(
            'torchaudio.functional.forced_align was removed in'
            ' torchaudio >=2.9'
            ' (see https://github.com/pytorch/audio/issues/3902)'
            f', pass {env_var}=1 to use custom implementation via recursion'
            ', but there may be problems with recursion limit'
        )
    elif len(true_tokens) == 0:
        # torchaudio.functional.forced_align will break here
        idx_per_frame = [blank_id] * len(log_probs)
        scores_per_frame = cast(list[float], log_probs[:, blank_id].tolist())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            # ignore deprecation warning
            alignments, scores = torchaudio.functional.forced_align( # type: ignore
                torch.tensor(log_probs).unsqueeze(0),
                torch.tensor(
                    true_tokens, dtype=torch.int32, device='cpu'
                ).unsqueeze(0),
                blank=blank_id,
            )
        idx_per_frame = cast(list[int], alignments[0].tolist()) # type: ignore
        scores_per_frame = cast(list[float], scores[0].tolist()) # type: ignore
    
    spans = [
        (value, start, end)
        for value, start, end in groupby_into_spans(idx_per_frame)
        if value != blank_id
    ]
    assert [value for value, _start, _end in spans] == true_tokens
    true_tokens_pos = [(start, end) for _value, start, end in spans]
    
    return idx_per_frame, scores_per_frame, true_tokens_pos


def forced_alignment_via_recursion(
    log_probs: FLOATS,
    true_tokens: list[int] | INTS,
    blank_id: int = 0,
) -> tuple[list[int], list[float]]:
    """Performs forced alignment via a custom recursive algorithm."""
    assert all(t != blank_id for t in true_tokens)
    
    @cache
    def _forced_alignment(
        log_probs_pos: int = 0,
        tokens_pos: int = 0,
        prev_token: int = blank_id,
    ) -> tuple[list[int], list[float]]:
        # Performs the forced alignment of log_probs[log_probs_pos:] to
        # tokens[tokens_pos:] given that prev_token_id was selected for the
        # frame log_probs_pos - 1. Will recursively solve via
        # _forced_alignment(log_probs_pos + 1, ...), considering several
        # options for the first frame
        
        if log_probs_pos >= len(log_probs):
            return [], [0 if tokens_pos >= len(true_tokens) else -np.inf]
        
        # each path is a list of token ids (possibly including blank_id) and
        # their log probabilities
        paths: list[tuple[list[int], list[float]]] = []
        
        # option 1: blank token is selected for the first frame
        tail_tokens, tail_p = _forced_alignment(
            log_probs_pos + 1, tokens_pos, blank_id
        )
        paths.append((
            [blank_id] + tail_tokens,
            [log_probs[log_probs_pos, blank_id]] + tail_p,
        ))
        
        # option 2: prev token is selected for the first frame (!= blank token)
        if prev_token != blank_id:
            tail_tokens, tail_p = _forced_alignment(
                log_probs_pos + 1, tokens_pos, prev_token
            )
            paths.append((
                [prev_token] + tail_tokens,
                [log_probs[log_probs_pos, prev_token]] + tail_p,
            ))
        
        # option 3: true_tokens[0] is selected for the first frame
        # (!= prev token)
        if (
            tokens_pos < len(true_tokens)
            and true_tokens[tokens_pos] != prev_token
        ):
            tail_tokens, tail_p = _forced_alignment(
                log_probs_pos + 1, tokens_pos + 1, true_tokens[tokens_pos]
            )
            paths.append((
                [true_tokens[tokens_pos]] + tail_tokens,
                [log_probs[log_probs_pos, true_tokens[tokens_pos]]] + tail_p,
            ))
        
        return max(paths, key=lambda path: sum(path[1]))
    
    result_tokens, result_p = _forced_alignment(0, 0, blank_id)
    if sum(result_p) == -np.inf:
        raise RuntimeError('cannot perform a force algnment')
    
    idx_per_frame = result_tokens
    scores_per_frame = result_p[:-1]  # drop the latest 0
    
    return idx_per_frame, scores_per_frame