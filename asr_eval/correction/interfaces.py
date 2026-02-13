from abc import ABC, abstractmethod

from asr_eval.utils.types import FLOATS


__all__ = [
    'TranscriptionCorrector',
]


class TranscriptionCorrector(ABC):
    """An abstract postprocessor capable of correcting ASR
    transcriptions.
    """
    
    @abstractmethod
    def correct(
        self, transcription: str, waveform: FLOATS | None = None
    ) -> str: ...