from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    'AudioSegment',
    'TimedText',
    'DiarizationSegment',
]


@dataclass(frozen=True)
class AudioSegment():
    '''
    An audio segment from .start_time to .end_time. Does not keep waveform or sampling_rate.
    '''
    start_time: float
    end_time: float
        
    def __post_init__(self):
        assert np.isfinite(self.start_time), f'Bad start time in the segment {self}'
        assert np.isfinite(self.end_time), f'Bad end time in the segment {self}'
        assert self.start_time >= 0, f'Start < 0 in the segment {self}'
        assert self.end_time >= 0, f'End < 0 in the segment {self}'
        assert self.start_time <= self.end_time, f'Start > end in the segment {self}'
    
    def start_pos(self, sampling_rate: int = 16_000) -> int:
        return int(self.start_time * sampling_rate)
    
    def end_pos(self, sampling_rate: int = 16_000) -> int:
        return int(self.end_time * sampling_rate)

    def slice(self, sampling_rate: int = 16_000) -> slice[int]:
        return slice(
            self.start_pos(sampling_rate=sampling_rate),
            self.end_pos(sampling_rate=sampling_rate),
        )
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def overlap_seconds(self, other: AudioSegment) -> float:
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        return max(0, overlap_end - overlap_start)


@dataclass(frozen=True)
class TimedText(AudioSegment):
    '''
    An AudioSegment with the corresponding text.
    '''
    text: str


@dataclass(frozen=True)
class DiarizationSegment(AudioSegment):
    '''
    An AudioSegment with the corresponding speaker index.
    '''
    speaker_idx: int