from itertools import pairwise
from typing import Literal

import numpy as np

from asr_eval.utils.misc import groupby_into_spans
from asr_eval.utils.types import FLOATS, INTS
from asr_eval.segments.segment import AudioSegment


__all__ = [
    'chunk_audio',
    'average_segment_features',
]


def chunk_audio(
    length: float,
    segment_length: float,
    segment_shift: float,
    last_chunk_mode: Literal['same_length', 'same_shift'] = 'same_length',
) -> list[AudioSegment]:
    """Chunks the audio uniformly.
    
    Args:
        length: A total audio length.
        segment_length: The desired length of each segment.
        segment_shift: The desired shift between conecutive segments.
    
    If :code:`length < segment_length`, returns a single chunk from 0 to
    length. Otherwise calculates how much chunks with the given
    :code:`segment_length` and :code:`segment_shift` fit into the
    :code:`length`. If the length does not accommodate an integer number
    of shifts, adds a single additional chunk:
    
    - If :code:`last_chunk_mode='same_length'`: from
      :code:`length - segment_length` to :code:`length`
    - If :code:`last_chunk_mode='same_shift'`: from
      :code:`<last_chunk_end> + segment_shift` to :code:`length`
    
    .. code-block:: none
    
        <---->  segment_shift
        <----------------------->  segment_length
        <--------------------------------------->  length
        =========================                |  
              ==========================         |  
                    ===========================  |  
                      ===========================|  # an additional chunk
                  
    Example:
        >>> chunk_audio(length=41, segment_length=30, segment_shift=5) # doctest: +NORMALIZE_WHITESPACE
        [AudioSegment(start_time=0.0, end_time=30.0),
        AudioSegment(start_time=5.0, end_time=35.0),
        AudioSegment(start_time=10.0, end_time=40.0),
        AudioSegment(start_time=11, end_time=41)]
    """
    
    assert length > 0 and segment_length > 0 and segment_shift > 0
    
    if length <= segment_length:
        segments = [(0, length)]
    else:
        segments = [
            (float(start), float(start) + segment_length)
            for start in np.arange(
                0, length - segment_length, step=segment_shift
            )
        ]
        last_start, last_end = segments[-1]
        if last_end < length:
            match last_chunk_mode:
                case 'same_length':
                    segments.append((length - segment_length, length))
                case 'same_shift':
                    if length > last_start + segment_shift:
                        segments.append((last_start + segment_shift, length))
    
    return [AudioSegment(start, end) for start, end in segments]


def merge_ctc_log_probs_by_blank_sep(
    segments: list[AudioSegment],
    log_probs: list[FLOATS],
    feature_tick_size: float,
    blank_id: int,
) -> FLOATS:
    """A merging algorithm for CTC log probs.
    
    Accepts CTC log prob matrices calculated on the given audio
    segments. The segments should have overlaps, but no triple overlaps:
    i-th segment end should occur after (i+1)-th segment start, but
    before (i+2)-th segment start.
    
    Merges log probs into a joint log probs matrix in the following way.
    When overlap occur between i-th and (i+1)-th matrix, searches for a
    reasonable time frame T where <blank> token is argmax for both
    matrices. If such a position was not found, sets T as a center of
    the overlap. In the result log probs before T are taken from i-th
    matrix, and log probs after T are taken from (i+1)-th matrix. 
    
    Args:
        segments: A list of segments. Typically obtained by a uniform
            chunking using
            :func:`~asr_eval.segments.chunking.chunk_audio`, but may
            also be non-uniform.
        features: CTC log probs array for each segment.
        feature_tick_size: A time interval between consecutive positions
            in :code:`features`.
        blank_id: Index of blank CTC token.
    """
    
    assert len(segments)
    
    tick_spans: list[tuple[int, int]] = []
    
    for segment, segment_features in zip(segments, log_probs, strict=True):
        feature_len = len(segment_features)
        tick_spans.append(_discretize(
            segment=segment,
            feature_tick_size=feature_tick_size,
            feature_len=feature_len,
        ))
    
    assert tick_spans[0][0] == 0, (
        f'Non-zero time frame for the first span: {tick_spans[0]}'
    )
        
    for span_idx in range(len(tick_spans) - 2):
        current_span = tick_spans[span_idx]
        next_span = tick_spans[span_idx + 1]
        next_next_span = tick_spans[span_idx + 2]
        assert current_span[1] >= next_span[0], (
            f'Empty space between spans {current_span} and {next_span}'
        )
        assert current_span[1] < next_next_span[0], (
            f'Triple overlap occur for {current_span}, {next_span}'
            f' and {next_next_span}'
        )
    
    # for each overlap, delta between overlap start
    # and separating position
    overlap_sizes: list[int] = []
    deltas: list[int] = []
    
    for (
        ((_start, end), current_log_probs),
        ((next_start, _next_end), next_log_probs),
    ) in pairwise(zip(tick_spans, log_probs)):
        overlap_size = end - next_start
        overlap_sizes.append(overlap_size)
        assert overlap_size >= 0
        
        if overlap_size == 0:
            delta = 0
        else:
            overlap_log_probs_1 = current_log_probs[-overlap_size:]
            overlap_log_probs_2 = next_log_probs[:overlap_size]
            blank_is_argmax_1 = overlap_log_probs_1.argmax(axis=1) == blank_id
            blank_is_argmax_2 = overlap_log_probs_2.argmax(axis=1) == blank_id
            blank_is_argmax_both = blank_is_argmax_1 & blank_is_argmax_2
            if not np.any(blank_is_argmax_both):
                delta = overlap_size // 2
            else:
                blanks = [
                    (i1, i2)
                    for value, i1, i2
                    in groupby_into_spans(blank_is_argmax_both)
                    if value
                ]
                best_blank_start, best_blank_end = (
                    max(blanks, key=lambda x: x[1] - x[0])
                )
                delta = (best_blank_end - 1 + best_blank_start) // 2
        
        deltas.append(delta)
    
    log_probs_to_concat: list[FLOATS] = []
    
    for span_idx, span_log_probs in enumerate(log_probs):
        prev_delta = deltas[span_idx - 1] if span_idx > 0 else 0
        cut_left = prev_delta
        if span_idx < len(log_probs) - 1:
            # not the last span
            overlap_size = overlap_sizes[span_idx]
            delta = deltas[span_idx]
            cut_right = overlap_size - delta
            log_probs_to_concat.append(span_log_probs[cut_left:-cut_right])
            
        else:
            # the last span
            log_probs_to_concat.append(span_log_probs[cut_left:])
    
    result = np.concatenate(log_probs_to_concat, axis=0)
    assert len(result) == tick_spans[-1][1]
    return result


def _discretize(
    segment: AudioSegment,
    feature_tick_size: float,
    feature_len: int,
) -> tuple[int, int]:
    """Given time interval and time frame size, returns start and end
    frame size. Calculates starts frame from the segment start time, and
    end frame from the start time plus feature_len. Asserts that end
    frame is close to the segmend end time with a tolerance of 2 frames
    """
    
    start_ticks = round(segment.start_time / feature_tick_size)
    expected_ticks = segment.duration / feature_tick_size
    assert abs(feature_len - expected_ticks) < 2, (
        f'Mismatch for segment {segment}:'
        f' expected {expected_ticks} ticks based'
        f' on the segment length and {feature_tick_size=},'
        f' got {feature_len} ticks in the'
        ' `features`. Incorrect feature_tick_size?'
    )
    return start_ticks, start_ticks + feature_len


def average_segment_features(
    segments: list[AudioSegment],
    features: list[FLOATS] | list[INTS],
    feature_tick_size: float,
    averaging_weights: Literal['beta', 'uniform'] = 'beta',
) -> FLOATS:
    """Given audio features calculated on the given audio chunking,
    averages them. The chunks (:code:`segments`) may overlap.
    
    Args:
        segments: A list of segments. Typically obtained by a uniform
            chunking using
            :func:`~asr_eval.segments.chunking.chunk_audio`, but may
            also be non-uniform.
        features: 2D feature array for each segment.
        feature_tick_size: A time interval between consecutive positions
            in :code:`features`.
        averaging_weights: May be "uniform" (flat) or "beta" (decaying
            at time edges of each feature array in :code:`features`). Is
            used to weight features.
    """
    import scipy
    
    assert len(segments)
    
    tick_spans: list[tuple[int, int]] = []
    weights: list[FLOATS] = []
    
    for segment, segment_features in zip(segments, features, strict=True):
        feature_len = len(segment_features)
        tick_spans.append(_discretize(
            segment=segment,
            feature_tick_size=feature_tick_size,
            feature_len=feature_len,
        ))
        
        match averaging_weights:
            case 'beta':
                w = 0.01 + scipy.stats.beta.pdf(
                    np.linspace(0, 1, num=feature_len), a=5, b=5
                )
            case 'uniform':
                w = np.ones(feature_len)
        weights.append(w)
    
    max_ticks = max(end for _start, end in tick_spans)
    
    sum_weights = np.zeros(max_ticks)
    for (start, end), segment_weights in zip(
        tick_spans, weights, strict=True
    ):
        sum_weights[start:end] += segment_weights
    
    averaged_features = np.zeros((max_ticks, features[0].shape[1]))
    for (start, end), segment_weights, segment_features in zip(
        tick_spans, weights, features, strict=True
    ):
        averaged_features[start:end] += (
            segment_features
            * (segment_weights / sum_weights[start:end])[:, None]
        )
    
    return averaged_features