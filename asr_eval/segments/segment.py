from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Self

import numpy as np


@dataclass(frozen=True)
class AudioSegment():
    """ An audio segment from :code:`.start_time` to :code:`.end_time`.
    
    Is immutable.
    """
    
    start_time: float
    end_time: float
        
    def __post_init__(self):
        assert np.isfinite(self.start_time), (
            f'Bad start time in the segment {self}'
        )
        assert np.isfinite(self.end_time), (
            f'Bad end time in the segment {self}'
        )
        assert self.start_time >= 0, (
            f'Start < 0 in the segment {self}'
        )
        assert self.end_time >= 0, (
            f'End < 0 in the segment {self}'
        )
        assert self.start_time <= self.end_time, (
            f'Start > end in the segment {self}'
        )
    
    def start_pos(self, sampling_rate: int = 16_000) -> int:
        """Get the start array position given a sampling rate."""
        
        return int(self.start_time * sampling_rate)
    
    def end_pos(self, sampling_rate: int = 16_000) -> int:
        """Get the end array position given a sampling rate."""
        
        return int(self.end_time * sampling_rate)

    def slice(self, sampling_rate: int = 16_000) -> slice[int]:
        """Get a :code:`slice` from the start to the end array position
        given a sampling rate.
        """
        
        return slice(
            self.start_pos(sampling_rate=sampling_rate),
            self.end_pos(sampling_rate=sampling_rate),
        )
    
    @property
    def duration(self) -> float:
        """A duration in seconds."""
        
        return self.end_time - self.start_time
    
    def overlap_seconds(self, other: AudioSegment) -> float:
        """An overlap with another segment in seconds."""
        
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        return max(0, overlap_end - overlap_start)

    def expand(self, left_indent: float, right_indent: float) -> Self:
        """Expands, given left and right indent. Avoids going into
        negative time positions. Returns a copy without modifying the
        original segment.
        """
        
        return replace(
            self,
            start_time=max(0, self.start_time - left_indent),
            end_time=self.end_time + right_indent,
        )
    
    def clip(self, max_sound_duration: float) -> Self:
        """Clips start and end times up to the given time. Returns a
        copy without modifying the original segment.
        """
        
        return replace(
            self,
            start_time=min(self.start_time, max_sound_duration),
            end_time=min(self.end_time, max_sound_duration),
        )
    
    @property
    def center_time(self):
        """Gets a center time in seconds."""
        
        return (self.start_time + self.end_time) / 2



@dataclass(frozen=True)
class TimedText(AudioSegment):
    """An :class:`~asr_eval.segments.AudioSegment` with the
    corresponding text.
    """
    
    text: str


@dataclass(frozen=True)
class DiarizationSegment(AudioSegment):
    """An :class:`~asr_eval.segments.AudioSegment` with the
    corresponding speaker index or name.
    """
    
    speaker: int | str


@dataclass(frozen=True, kw_only=True)
class TimedDiarizationText(TimedText, DiarizationSegment):
    pass