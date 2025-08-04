from typing import Any, override
from pisets import Pisets

from asr_eval.segments.segment import TimedText

from .base.interfaces import TimedTranscriber
from ..utils.types import FLOATS


class PisetsWrapper(TimedTranscriber):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.pisets = Pisets(**self.kwargs)
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        return [
            TimedText(
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.whisper_text,
            )
            for seg in self.pisets(waveform)
        ]