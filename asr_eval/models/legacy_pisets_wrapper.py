from pathlib import Path
import sys
from typing import override, cast

from transformers import Pipeline

from ..segments.segment import TimedText
from .base.interfaces import TimedTranscriber
from ..utils.types import FLOATS


__all__ = [
    'LegacyPisetsWrapper',
]


class LegacyPisetsWrapper(TimedTranscriber):
    '''
    A Pisets transcriber from
    https://github.com/bond005/pisets/tree/e095ae626bbd18bb4490b9745d0acc34006c4eb8
    
    Requires a manual cloning into the `repo_dir`.
    '''
    def __init__(
        self,
        repo_dir: str | Path,
        min_segment_size: int = 1,
        max_segment_size: int = 20,
        use_vad: bool = False,
    ):
        self.repo_dir = repo_dir
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size
        self.use_vad = use_vad
        
        sys.path.append(str(self.repo_dir))
        from asr.asr import ( # type: ignore
            transcribe, # type: ignore
            initialize_model_for_speech_segmentation, # type: ignore
            initialize_model_for_speech_classification, # type: ignore
            initialize_model_for_speech_recognition, # type: ignore
        )
        
        self.segmenter: Pipeline = initialize_model_for_speech_segmentation('ru')
        if self.use_vad:
            self.voice_activity_detector: Pipeline = initialize_model_for_speech_classification()
        self.asr: Pipeline = initialize_model_for_speech_recognition('ru')
        self.pisets_transcribe = transcribe # type: ignore
        
        sys.path.pop()
    
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        segments = cast(list[tuple[float, float, str]], self.pisets_transcribe( # type: ignore
            mono_sound=waveform, # type: ignore
            segmenter=self.segmenter, # type: ignore
            voice_activity_detector=( # type: ignore
                self.voice_activity_detector
                if self.use_vad
                else lambda _waveform: [{'label': 'speech'}]), # type: ignore
            asr=self.asr, # type: ignore
            min_segment_size=self.min_segment_size, # type: ignore
            max_segment_size=self.max_segment_size, # type: ignore
        ))
        return [TimedText(start, end, text) for start, end, text in segments]