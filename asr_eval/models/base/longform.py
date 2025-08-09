import re
from typing import Literal, override

from ...segments.chunking import average_segment_features, chunk_audio
from .interfaces import CTC, TimedTranscriber, Transcriber, Segmenter, ContextualTranscriber
from ...segments.segment import TimedText
from ...utils.types import FLOATS


__all__ = [
    'LongformVAD',
    'LongformCTC',
    'ContextualLongformVAD',
]


class LongformVAD(TimedTranscriber):
    '''
    A longform wrapper for a shortform model. Uses a given segmenter, such as
    PyannoteSegmenter(), to segment into chunks, then applies a shortform model
    to each chunk independently.
    
    If a shortform model is a TimedTranscriber, concatenates TimedText lists
    for all chunks.
    '''
    def __init__(
        self,
        shortform_model: Transcriber,
        segmenter: Segmenter,
        min_sec: float = 30,
    ):
        self.shortform_model = shortform_model
        self.segmenter = segmenter
        self.min_sec = min_sec
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        audio_len = len(waveform) / 16_000
        if audio_len < self.min_sec:
            if isinstance(self.shortform_model, TimedTranscriber):
                return self.shortform_model.timed_transcribe(waveform)
            else:
                return [TimedText(0, audio_len, self.shortform_model.transcribe(waveform))]
        else:
            results: list[TimedText] = []
            for segment in self.segmenter(waveform):
                if isinstance(self.shortform_model, TimedTranscriber):
                    results += self.shortform_model.timed_transcribe(waveform[segment.slice()])
                else:
                    results.append(TimedText(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=self.shortform_model.transcribe(waveform[segment.slice()]),
                    ))
            
            return results


class LongformCTC(CTC):
    '''
    A wrapper to apply a short-form CTC model to a long-form audio. Segments audio uniformly with
    overlaps, then averages the logprobs for all segments.
    
    By default averages with beta-distributed weights (averaging_weights='beta'), since a model may
    be less certain on the edges of the segment.
    '''
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


class ContextualLongformVAD(TimedTranscriber):
    '''
    Is similar LongformVAD, but each time passes the previously transcribed text, up
    to the `max_history_words`, as a context for the next chunk when transcribing it.
    
    Requies a shortform model to be a ContextualTranscriber.
    '''
    def __init__(
        self,
        shortform_model: ContextualTranscriber,
        segmenter: Segmenter,
        pass_history: bool = True,
        max_history_words: int | None = 100,
        min_sec: float = 30,
    ):
        self.shortform_model = shortform_model
        self.segmenter = segmenter
        self.pass_history = pass_history
        self.max_history_words = max_history_words
        self.min_sec = min_sec
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        audio_len = len(waveform) / 16_000
        if audio_len < self.min_sec:
            return [TimedText(0, audio_len, self.shortform_model.transcribe(waveform))]
        else:
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