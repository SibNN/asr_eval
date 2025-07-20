from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioSegment():
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