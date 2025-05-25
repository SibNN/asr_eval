from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Adopted from https://docs.pytorch.org/audio/2.2.0/tutorials/forced_alignment_tutorial.html


def force_alignment(
    log_probs: npt.NDArray[np.floating],
    true_tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> tuple[list[Segment], list[Point], npt.NDArray[np.floating]]:
    '''
    Performs a force alignment of token probabilities (from CTC head) and a given token sequence.
    Need to specify blank ID token correctly.
    
    NOTE: For Gigaam use blank_id=model.decoding.blank_id
    
    Each output segment represents a time span and score for a voken from model vocabulary. The other
    two return values represents points and trellis, to use in plot_alignments or plot_trellis_with_path.
    
    See example in tests/test_force_alignment.py
    ```
    '''
    trellis = get_trellis(log_probs, true_tokens, blank_id)
    path = backtrack(trellis, log_probs, true_tokens)
    segments = merge_repeats(path)
    assert len(segments) == len(true_tokens)
    for i in range(len(true_tokens)):
        segments[i].token = true_tokens[i]
    return segments, path, trellis


def plot_alignments(
    segments: list[Segment],
    trellis: npt.NDArray[np.floating],
    vocab: list[str] | None = None,
):
    plt.figure(figsize=(10, 3)) # type: ignore

    plt.imshow(trellis.T, origin="lower", aspect="auto") # type: ignore
    plt.xticks([]) # type: ignore
    plt.yticks([]) # type: ignore

    for i, seg in enumerate(segments):
        plt.annotate( # type: ignore
            vocab[seg.token] if vocab else str(seg.token),
            (seg.start, i - 0.2), size="small"
        )
    
    plt.show() # type: ignore


def plot_trellis_with_path(
    trellis: npt.NDArray[np.floating],
    path: list[Point],
):
    trellis_with_path = np.copy(trellis)
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_seq_index] = np.nan
    plt.imshow(trellis_with_path.T, origin="lower") # type: ignore
    plt.title("The path found by backtracking") # type: ignore
    plt.tight_layout()
    plt.show() # type: ignore


def get_trellis(
    log_probs: npt.NDArray[np.floating],
    tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> npt.NDArray[np.floating]:
    trellis = np.zeros((len(log_probs), len(tokens)), dtype=log_probs.dtype)
    trellis[1:, 0] = np.cumsum(log_probs[1:, blank_id])
    trellis[0, 1:] = -np.inf
    trellis[-len(tokens) + 1:, 0] = np.inf

    for t in range(len(log_probs) - 1):
        trellis[t + 1, 1:] = np.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + log_probs[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + log_probs[t, tokens[1:]],
        )
    return trellis


@dataclass
class Point:
    token_seq_index: int
    time_index: int
    score: float


def backtrack(
    trellis: npt.NDArray[np.floating],
    log_probs: npt.NDArray[np.floating],
    tokens: list[int] | npt.NDArray[np.integer],
    blank_id: int = 0,
) -> list[Point]:
    t, j = trellis.shape[0] - 1, trellis.shape[1] - 1

    path = [Point(j, t, float(np.exp(log_probs[t, blank_id])))]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = log_probs[t - 1, blank_id]
        p_change = log_probs[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = np.exp(p_change if changed > stayed else p_stay)
        path.append(Point(j, t, float(prob)))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = np.exp(log_probs[t - 1, blank_id])
        path.append(Point(j, t - 1, float(prob)))
        t -= 1

    return path[::-1]
   
    
@dataclass
class Segment:
    token: int
    start: int
    end: int
    score: float

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path: list[Point]) -> list[Segment]:
    i1, i2 = 0, 0
    segments: list[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_seq_index == path[i2].token_seq_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                token=-1,  # unknown for now, will set later
                start=path[i1].time_index,
                end=path[i2 - 1].time_index + 1,
                score=score,
            )
        )
        i1 = i2
    return segments