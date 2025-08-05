from abc import ABC, abstractmethod


class TranscriptionCorrector(ABC):
    @abstractmethod
    def correct(self, transcription: str) -> str: ...