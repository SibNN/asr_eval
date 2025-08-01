from typing import Any, override
from pisets import Pisets

from .base import Transcriber
from ..utils.types import FLOATS


class PisetsWrapper(Transcriber):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.pisets: Pisets | None = None
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        self.pisets = self.pisets or Pisets(**self.kwargs) # type: ignore
        return ' '.join(seg.whisper_text for seg in self.pisets(waveform))