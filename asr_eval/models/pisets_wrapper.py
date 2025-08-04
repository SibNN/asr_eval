from typing import Any, override
from pisets import Pisets

from .base.interfaces import Transcriber
from ..utils.types import FLOATS


class PisetsWrapper(Transcriber):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.pisets = Pisets(**self.kwargs)
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return ' '.join(seg.whisper_text for seg in self.pisets(waveform))