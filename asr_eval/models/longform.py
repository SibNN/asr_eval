import re
from typing import Literal, override

from ..segments.chunking import average_segment_features, chunk_audio
from .base import CTC, TimedTranscriber, Transcriber, Segmenter, ContextualTranscriber
from ..segments.segment import TimedText
from ..utils.types import FLOATS


class LongformTranscriberVAD(TimedTranscriber):
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


class LongformCTC(CTC):
    def __init__(
        self,
        shortform_model: CTC,
        segment_length: float = 30,
        segment_shift: float = 10,
        averaging_weights: Literal['beta', 'uniform'] = 'beta',
    ):
        self.shortform_model = shortform_model
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.averaging_weights: Literal['beta', 'uniform'] = averaging_weights
    
    @override
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        return [self._merge_log_probs(waveform) for waveform in waveforms]
    
    @property
    @override
    def blank_id(self) -> int:
        return self.shortform_model.blank_id
    
    @property
    @override
    def tick_size(self) -> float:
        return self.shortform_model.tick_size
    
    @override
    def decode(self, token: int) -> str:
        return self.shortform_model.decode(token)
    
    def _merge_log_probs(self, waveform: FLOATS) -> FLOATS:
        segments = chunk_audio(
            len(waveform) / 16_000,
            segment_length=self.segment_length,
            segment_shift=self.segment_shift,
        )
        log_probs = [
            # TODO batching
            self.shortform_model.ctc_log_probs([waveform[segment.slice()]])[0]
            for segment in segments
        ]
        merged_log_probs = average_segment_features(
            segments=segments,
            features=log_probs,
            feature_tick_size=self.tick_size,
            averaging_weights=self.averaging_weights,
        )
        return merged_log_probs


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
        for segment in segments:
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