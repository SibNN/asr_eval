import re
from typing import override

from .base import TimedTranscriber, Transcriber, Segmenter, ContextualTranscriber
from ..segments.segment import TimedText
from ..utils.types import FLOATS


class LongformTranscriber(TimedTranscriber):
    def __init__(
        self,
        shortform_model: Transcriber,
        segmenter: Segmenter,
    ):
        self.shortform_model = shortform_model
        self.segmenter = segmenter
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        return [
            TimedText(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=self.shortform_model.transcribe(waveform[segment.slice()]),
            )
            for segment in self.segmenter(waveform)
        ]


class ContextualLongformTranscriber(TimedTranscriber):
    def __init__(
        self,
        shortform_model: ContextualTranscriber,
        segmenter: Segmenter,
        pass_history: bool = True,
        max_history_words: int | None = 100,
    ):
        self.shortform_model = shortform_model
        self.segmenter = segmenter
        self.pass_history = pass_history
        self.max_history_words = max_history_words
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        segments = self.segmenter(waveform)
        
        transcriptions: list[TimedText] = []
        for i, segment in enumerate(segments):
            history = ' '.join(t.text for t in transcriptions)
            if self.max_history_words is not None:
                words = list(re.finditer(r'\w+', history))
                if len(words) > self.max_history_words:
                    first_word = words[-self.max_history_words]
                    history = history[first_word.start():]
                
            text = self.shortform_model.contextual_transcribe(
                waveform[segment.slice()], prev_transcription=history,
            )
            
            transcriptions.append(TimedText(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=text,
            ))
            
        return transcriptions