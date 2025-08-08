from abc import ABC, abstractmethod
from typing import cast, override

from ...ctc.base import ctc_mapping
from ...segments.segment import AudioSegment, TimedText
from ...utils.types import FLOATS


__all__ = [
    'Segmenter',
    'Transcriber',
    'TimedTranscriber',
    'CTC',
    'ContextualTranscriber',
]


class Segmenter(ABC):
    '''
    Segments a long-form audio into chunks.
    
    Any parameters, such as max segment size, should go into a class constructor.
    '''
    @abstractmethod
    def __call__(self, waveform: FLOATS) -> list[AudioSegment]: ...


class Transcriber(ABC):
    '''
    A transcriber (audio -> text) to evaluate on any dataset.
    '''
    @abstractmethod
    def transcribe(self, waveform: FLOATS) -> str: ...


class TimedTranscriber(Transcriber):
    '''
    A timed transcriber (audio -> timed text chunks) to evaluate on any dataset.
    
    Given timed text chunks, the trancscription by default is generated
    by concatenating them with space. Subclasses may override this.
    '''
    @abstractmethod
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]: ...
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return ' '.join(x.text for x in self.timed_transcribe(waveform))


class CTC(Transcriber):
    '''
    Converts audio into CTC log probs (should sum up to 1).
    '''
    @abstractmethod
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        '''
        Calculates log probs (should sum up to 1)
        '''
        ...
    
    @property
    @abstractmethod
    def blank_id(self) -> int:
        '''
        Vocabulary index for <blank> CTC token
        '''
        ...
    
    @property
    @abstractmethod
    def tick_size(self) -> float:
        '''
        Time interval in seconds between consecutive time steps in log probs matrix
        '''
        ...
    
    @abstractmethod
    def decode(self, token: int) -> str:
        '''
        Decode a single token index into string (usually a single letter)
        
        Note that this does not support Whisper-style BPE encoding: each single token
        should be decoded into a valid unicode string.
        '''
        ...
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # converts CTC log probs into the output text via argmax
        log_probs = self.ctc_log_probs([waveform])[0]
        labels = cast(list[int], log_probs.argmax(axis=-1, keepdims=False).tolist())
        tokens = ctc_mapping(labels, self.blank_id)
        return ''.join(self.decode(t) for t in tokens)


class ContextualTranscriber(Transcriber):
    '''
    A transcriber being able to accept previous transcription as a context.
    '''
    @abstractmethod
    def contextual_transcribe(
        self, waveform: FLOATS, prev_transcription: str = ''
    ) -> str: ...
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return self.contextual_transcribe(waveform)