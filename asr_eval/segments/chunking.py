from typing import Literal

import numpy as np
import scipy

from ..utils.types import FLOATS, INTS
from .segment import AudioSegment


def chunk_audio(
    length: float,
    segment_length: float,
    segment_shift: float,
    last_chunk_mode: Literal['same_length', 'same_shift'] = 'same_length',
) -> list[AudioSegment]:
    '''
    Arguments:
    - length: a total audio length
    - segment_length: the desired length of each segment
    - segment_shift: the desired shift between conecutive segments
    
    If length < segment_length, returns a single chunk from 0 to length.
    
    Otherwise calculates how much chunks with the given `segment_length` and `segment_shift`
    fit into the `length`. If the length does not accommodate an integer number of shifts,
    adds a single additional chunk:
    - If last_chunk_mode='same_length': from `length - segment_length` to `length`
    - If last_chunk_mode='same_shift': from `<last_chunk_end> + segment_shift` to `length`
    
    <---->  segment_shift
    <----------------------->  segment_length
    <--------------------------------------->  length
    =========================                |  
          ==========================         |  
                ===========================  |  
                  ===========================|  # an additional chunk
                  
    Example: `chunk_audio(length=41, segment_length=30, segment_shift=5)`
    >>> [AudioSegment(start_time=0.0, end_time=30.0),
         AudioSegment(start_time=5.0, end_time=35.0),
         AudioSegment(start_time=10.0, end_time=40.0),
         AudioSegment(start_time=11, end_time=41)]
    '''
    assert length > 0 and segment_length > 0 and segment_shift > 0
    
    if length <= segment_length:
        segments = [(0, length)]
    else:
        segments = [
            (float(start), float(start) + segment_length)
            for start in np.arange(0, length - segment_length, step=segment_shift)
        ]
        _last_start, last_end = segments[-1]
        if last_end < length:
            match last_chunk_mode:
                case 'same_length':
                    segments.append((length - segment_length, length))
                case 'same_shift':
                    if length > last_end + segment_shift:
                        segments.append((last_end + segment_shift, length))
    
    return [AudioSegment(start, end) for start, end in segments]


def average_segment_features(
    segments: list[AudioSegment],
    features: list[FLOATS] | list[INTS],
    feature_tick_size: float,
    averaging_weights: Literal['beta', 'uniform'] = 'beta',
) -> FLOATS:
    '''
    
    '''
    assert len(segments)
    
    tick_spans: list[tuple[int, int]] = []
    weights: list[FLOATS] = []
    
    for segment_idx, (segment, segment_features) in enumerate(zip(segments, features, strict=True)):
        start_ticks = round(segment.start_time / feature_tick_size)
        actual_ticks = len(segment_features)
        expected_ticks = segment.duration / feature_tick_size
        assert abs(actual_ticks - expected_ticks) < 1.1, (
            f'Mismatch for segment #{segment_idx} {segment}:'
            f' expected {expected_ticks} ticks based on the segment length and {feature_tick_size=},'
            f' got {actual_ticks} ticks in the `features`. Incorrect feature_tick_size?'
        )
        tick_spans.append((start_ticks, start_ticks + actual_ticks))
        
        match averaging_weights:
            case 'beta':
                w = 0.01 + scipy.stats.beta.pdf(np.linspace(0, 1, num=actual_ticks), a=5, b=5)
            case 'uniform':
                w = np.ones(actual_ticks)
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
            segment_features * (segment_weights / sum_weights[start:end])[:, None]
        )
    
    return averaged_features