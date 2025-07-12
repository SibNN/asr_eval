from typing import Any, override
from pisets import Pisets

from asr_eval.models.base import ASREvalWrapper, AUDIO_TYPE


class PisetsWrapper(ASREvalWrapper):
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.pisets: Pisets | None = None
    
    @override
    def __call__(self, waveforms: list[AUDIO_TYPE]) -> list[str]:
        self.pisets = self.pisets or Pisets(**self.kwargs) # type: ignore
        return [
            ' '.join(seg.whisper_text for seg in self.pisets(waveform))
            for waveform in waveforms
        ]