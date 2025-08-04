from pathlib import Path
import sys
from typing import override

from transformers import Pipeline

from .base.interfaces import Transcriber
from ..utils.types import FLOATS


class LegacyPisetsWrapper(Transcriber):
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
        self.transcribe = transcribe # type: ignore
        
        sys.path.pop()
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        segments: list[tuple[float, float, str]] = self.transcribe( # type: ignore
            mono_sound=waveform, # type: ignore
            segmenter=self.segmenter, # type: ignore
            voice_activity_detector=( # type: ignore
                self.voice_activity_detector
                if self.use_vad
                else lambda _waveform: [{'label': 'speech'}]), # type: ignore
            asr=self.asr, # type: ignore
            min_segment_size=self.min_segment_size, # type: ignore
            max_segment_size=self.max_segment_size, # type: ignore
        )
        return ' '.join(text for _start, _end, text in segments) # type: ignore