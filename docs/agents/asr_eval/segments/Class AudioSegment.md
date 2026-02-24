# Class AudioSegment (defined in asr_eval/segments/segment.py at lines 8-96)

@dataclasses.dataclass(frozen=True)
class AudioSegment():
    """ An audio segment from :code:`.start_time` to :code:`.end_time`.

    Is immutable.
    """
    ...

    start_time: float

    end_time: float

    def start_pos(self, sampling_rate: int = 16_000) -> int:
        """Get the start array position given a sampling rate."""
        ...

    def end_pos(self, sampling_rate: int = 16_000) -> int:
        """Get the end array position given a sampling rate."""
        ...

    def slice(self, sampling_rate: int = 16_000) -> slice[int]:
        """Get a :code:`slice` from the start to the end array position
        given a sampling rate.
        """
        ...

    @property
    def duration(self) -> float:
        """A duration in seconds."""
        ...

    def overlap_seconds(self, other: asr_eval.segments.segment.AudioSegment) -> float:
        """An overlap with another segment in seconds."""
        ...

    def expand(self, left_indent: float, right_indent: float) -> typing.Self:
        """Expands, given left and right indent. Avoids going into
        negative time positions. Returns a copy without modifying the
        original segment.
        """
        ...

    def clip(self, max_sound_duration: float) -> typing.Self:
        """Clips start and end times up to the given time. Returns a
        copy without modifying the original segment.
        """
        ...

    @property
    def center_time(self):
        """Gets a center time in seconds."""
        ...