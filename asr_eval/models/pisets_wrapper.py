from typing import Any, override
from pisets import Pisets

from .base import ASREvalWrapper
from ..utils.types import FLOATS


class PisetsWrapper(ASREvalWrapper):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.pisets: Pisets | None = None
    
    @override
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        self.pisets = self.pisets or Pisets(**self.kwargs) # type: ignore
        return [
            ' '.join(seg.whisper_text for seg in self.pisets(waveform))
            for waveform in waveforms
        ]