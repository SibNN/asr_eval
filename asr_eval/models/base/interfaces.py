from abc import ABC, abstractmethod
from typing import cast, override

from ...ctc.base import ctc_mapping
from ...segments.segment import AudioSegment, TimedText
from ...utils.types import FLOATS


class Segmenter(ABC):
    @abstractmethod
    def __call__(self, waveform: FLOATS) -> list[AudioSegment]: ...


class Transcriber(ABC):
    '''
    A transcriber (audio -> text) to evaluate on any dataset.
    '''
    @abstractmethod
    def transcribe(self, waveform: FLOATS) -> str: ...


class TimedTranscriber(Transcriber):
    @abstractmethod
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]: ...
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        return ' '.join(x.text for x in self.timed_transcribe(waveform))


class CTC(Transcriber):
    '''
    Converts audio into CTC log probas (should sum up to 1)
    '''
    @abstractmethod
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]: ...
    
    @property
    @abstractmethod
    def blank_id(self) -> int: ...
    
    @property
    @abstractmethod
    def tick_size(self) -> float: ...
    
    @abstractmethod
    def decode(self, token: int) -> str: ...
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
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





# class ASREvalWrapper(ABC):
#     @abstractmethod
#     def __call__(self, waveforms: list[FLOATS]) -> list[str]:
#         ...


# class AbstractShortformAudioLLM(ASREvalWrapper):
#     '''
#     An example (TODO delete?)
#     accepts `prev_transcription` and `domain_words` into `transcribe`
#     '''
#     @override
#     def transcribe(
#         self,
#         waveform: FLOATS,
#         prev_transcription: str | None = None,
#         domain_words: str | None = None,
#         **kwargs: Any,
#     ) -> list[TimedText]:
#         prompt = 'Recognize the audio {{audio}}.'
#         if prev_transcription:
#             prompt += f' The previous transcription was: {prev_transcription}.'
#         if domain_words:
#             prompt += f' The following text may be related: {domain_words}.'
#         text = self.run_audio_llm_inference(prompt, waveform)
#         return [TimedText(0, len(waveform) / 16_000, text)]
    
#     def run_audio_llm_inference(self, prompt: str, waveform: FLOATS) -> str:
#         raise NotImplementedError()