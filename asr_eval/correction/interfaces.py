from abc import ABC, abstractmethod

from ..utils.types import FLOATS


__all__ = [
    'TranscriptionCorrector',
]


class TranscriptionCorrector(ABC):
    '''
    Any post-processor capable of correcting ASR transcriptions.
    '''
    @abstractmethod
    def correct(self, transcription: str, waveform: FLOATS | None = None) -> str: ...