from abc import ABC, abstractmethod
from typing import cast, override

from asr_eval.ctc.base import ctc_mapping
from asr_eval.segments.segment import AudioSegment, TimedText
from asr_eval.utils.types import FLOATS


__all__ = [
    'Segmenter',
    'Transcriber',
    'TimedTranscriber',
    'CTC',
    'ContextualTranscriber',
]


class Segmenter(ABC):
    """An abstract model that segments a long-form audio into chunks
    containing speech.
    
    Any parameters, such as max segment size, should go into a class
    constructor.
    """
    
    @abstractmethod
    def __call__(self, waveform: FLOATS) -> list[AudioSegment]:
        """Segments a float32 waveform, typically normalized
        from -1 to 1.
        """


class Transcriber(ABC):
    """An abstract transcriber (audio -> text) to evaluate on any
    dataset.
    """
    
    @abstractmethod
    def transcribe(self, waveform: FLOATS) -> str:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1.
        """


class TimedTranscriber(Transcriber):
    """An abstract timed transcriber (audio -> timed text chunks) to
    evaluate on any dataset.
    
    Overrides a
    :meth:`~asr_eval.models.base.interfaces.Transcriber.transcribe`
    method by concatenating the test chunks by space. Subclasses may
    custoimize this.
    """
    
    @abstractmethod
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1, into a list of texts with timings. Typically
        the texts are to be concatenated via space, so leading or
        trailing spaces in each chunk are not required.
        """
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return ' '.join(x.text for x in self.timed_transcribe(waveform))


class CTC(Transcriber):
    """An abstract CTC model that converts audio into log probabilties
    for each time frame.
    
    Implementations override a list of additional methods for additional
    actions such as decoding.
    """
    
    @abstractmethod
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        """Calculates log probabilties each time frame, given a float32
        waveform, typically normalized from -1 to 1. Exponent from the
        log probabilties should sum up to 1 for each time frame.
        
        Typically obtained from logits via
        :code:`torch.nn.functional.log_softmax`. Note that the returned
        value should be a numpy array, not a torch tensor.
        """
    
    @property
    @abstractmethod
    def blank_id(self) -> int:
        """An index in vocabulary for <blank> CTC token."""
    
    @property
    @abstractmethod
    def tick_size(self) -> float:
        """A time interval in seconds between consecutive time frames in
        the log probs matrix.
        """
    
    @property
    @abstractmethod
    def vocab(self) -> tuple[str, ...]:
        """Returns a vocabulary: a character (usually a single letter)
        or character sequence for each vocabulary index, or empty string
        for blank token.
        
        Note that this does not fully support Whisper-style BPE
        encoding: each single token should correspond to a valid unicode
        string.
        """
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # converts CTC log probs into the output text via argmax
        log_probs = self.ctc_log_probs([waveform])[0]
        labels = cast(
            list[int],
            log_probs.argmax(axis=-1, keepdims=False).tolist()
        )
        tokens = ctc_mapping(labels, self.blank_id)
        vocab = self.vocab
        return ''.join(vocab[t] for t in tokens)


class ContextualTranscriber(Transcriber):
    """An abstract transcriber being able to accept previous
    transcription as a context.
    """
    
    @abstractmethod
    def contextual_transcribe(
        self, waveform: FLOATS, prev_transcription: str = ''
    ) -> str:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1. The :code:`prev_transcription` represents a
        transcription from all the previous text before the current
        :code:`waveform`.
        """
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return self.contextual_transcribe(waveform)