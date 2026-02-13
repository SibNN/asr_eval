from collections.abc import Callable
from dataclasses import replace
import re
from typing import Literal, override

from asr_eval.segments.chunking import average_segment_features, chunk_audio, merge_ctc_log_probs_by_blank_sep
from asr_eval.models.base.interfaces import (
    CTC, TimedTranscriber, Transcriber, Segmenter, ContextualTranscriber
)
from asr_eval.segments.segment import TimedText
from asr_eval.utils.types import FLOATS


__all__ = [
    'LongformVAD',
    'LongformCTC',
    'ContextualLongformVAD',
]


class LongformVAD(TimedTranscriber):
    """A longform transcriber wrapper for any shortform model.
    
    Longform transcriber means one being able to transcribe long audios.
    The concrete threshold between "long" and "short" audio may be
    specific for the provided :code:`shortform_model`.
    
    The current wrapper uses a provided segmenter to segment into
    chunks, then applies a shortform model to each chunk independently.
    If a shortform model is a
    :class:`~asr_eval.models.base.interfaces.TimedTranscriber`,
    concatenates the resulting lists for all chunks, while correcting
    the timestamps to be relative to the whole audio.
    
    Example:
        >>> # requires `pip install pyannote.audio>=4` for `PyannoteSegmenter`
        >>> from asr_eval.models.base.longform import LongformVAD
        >>> from asr_eval.models.pyannote_vad import PyannoteSegmenter
        >>> from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper
        >>> LongformVAD(  #doctest: +SKIP
        ...     Wav2vec2Wrapper('facebook/wav2vec2-base-960h'),
        ...     PyannoteSegmenter()
        ... )
    
    See also: :class:`~asr_eval.models.base.longform.LongformCTC`.
    """
    
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
                seg = TimedText(
                    0, audio_len, self.shortform_model.transcribe(waveform)
                )
                return [seg]
        else:
            results: list[TimedText] = []
            for segment in self.segmenter(waveform):
                if isinstance(self.shortform_model, TimedTranscriber):
                    outputs = self.shortform_model.timed_transcribe(
                        waveform[segment.slice()]
                    )
                    delta = segment.start_time
                    results += [
                        replace(
                            seg,
                            start_time=seg.start_time + delta,
                            end_time=seg.end_time + delta,
                        )
                        for seg in outputs
                    ]
                else:
                    results.append(TimedText(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=self.shortform_model.transcribe(
                            waveform[segment.slice()]
                        ),
                    ))
            
            return results


class LongformCTC(CTC):
    """A wrapper to apply a shortform CTC model to a longform audio.
    
    Longform transcriber means one being able to transcribe long audios.
    The current wrapper segments audio uniformly with overlaps, then
    averages the logprobs for all segments. By default averages with
    beta-distributed weights (:code:`averaging_weights='beta'`), because
    a model may be less certain on the edges of the segment.
    
    See also: :class:`~asr_eval.models.base.longform.LongformVAD`.
    """
    
    def __init__(
        self,
        shortform_model: CTC,
        segment_length: float = 30,
        segment_shift: float = 10,
        averaging_weights: Literal['beta', 'uniform', 'blank_sep'] = 'beta',
        pbar_callback: Callable[[int, int], None] | None = None,
    ):
        self.shortform_model = shortform_model
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.averaging_weights: (
            Literal['beta', 'uniform', 'blank_sep']
        ) = averaging_weights
        self.pbar_callback = pbar_callback
    
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
    
    @property
    @override
    def vocab(self) -> tuple[str, ...]:
        return self.shortform_model.vocab
    
    def _merge_log_probs(self, waveform: FLOATS) -> FLOATS:
        segments = chunk_audio(
            len(waveform) / 16_000,
            segment_length=self.segment_length,
            segment_shift=self.segment_shift,
            last_chunk_mode=(
                'same_length'
                if self.averaging_weights != 'blank_sep'
                # use same shift to avoid possible triple overlap
                else 'same_shift'
            )
        )
        
        log_probs: list[FLOATS] = []
        for segment_idx, segment in enumerate(segments):
            # TODO batching
            waveform_part = waveform[segment.slice()]
            log_probs.append(
                self.shortform_model.ctc_log_probs([waveform_part])[0]
            )
            if self.pbar_callback is not None:
                self.pbar_callback(segment_idx, len(segments))
        
        if self.averaging_weights == 'blank_sep':
            merged_log_probs = merge_ctc_log_probs_by_blank_sep(
                segments=segments,
                log_probs=log_probs,
                feature_tick_size=self.tick_size,
                blank_id=self.blank_id,
            )
        else:
            merged_log_probs = average_segment_features(
                segments=segments,
                features=log_probs,
                feature_tick_size=self.tick_size,
                averaging_weights=self.averaging_weights,
            )
        
        return merged_log_probs


class ContextualLongformVAD(TimedTranscriber):
    """A wrapper that is similar to
    :class:`~asr_eval.models.base.longform.LongformVAD`, but for each
    chunk passes the previously transcribed text, up to the
    :code:`max_history_words`, as a context for the next chunk when
    transcribing it.
    
    Requies a shortform model to be a
    :class:`~asr_eval.models.base.interfaces.ContextualTranscriber`.
    """
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
            seg = TimedText(
                0, audio_len, self.shortform_model.transcribe(waveform)
            )
            return [seg]
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