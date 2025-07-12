from pathlib import Path
import sys
from typing import override

from transformers import Pipeline

from asr_eval.models.base import AUDIO_TYPE, ASREvalWrapper


class LegacyPisetsWrapper(ASREvalWrapper):
    def __init__(
        self,
        repo_dir: str | Path,
        min_segment_size: int = 1,
        max_segment_size: int = 20,
        use_vad: bool = False,
    ):
        self.repo_dir = repo_dir
        self.segmenter: Pipeline | None = None
        self.voice_activity_detector: Pipeline | None = None
        self.asr: Pipeline | None = None
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size
        self.use_vad = use_vad
    
    def instantiate(self):
        sys.path.append(str(self.repo_dir))
        from asr.asr import ( # type: ignore
            transcribe, # type: ignore
            initialize_model_for_speech_segmentation, # type: ignore
            initialize_model_for_speech_classification, # type: ignore
            initialize_model_for_speech_recognition, # type: ignore
        )
        self.segmenter = initialize_model_for_speech_segmentation('ru')
        if self.use_vad:
            self.voice_activity_detector = initialize_model_for_speech_classification()
        self.asr = initialize_model_for_speech_recognition('ru')
        self.transcribe = transcribe # type: ignore
        sys.path.pop()
    
    @override
    def __call__(self, waveforms: list[AUDIO_TYPE]) -> list[str]:
        if self.segmenter is None:
            self.instantiate()
        assert self.segmenter is not None
        assert self.asr is not None
        
        texts: list[str] = []
        for waveform in waveforms:
            segments: list[tuple[float, float, str]] = self.transcribe( # type: ignore
                mono_sound=waveform,
                segmenter=self.segmenter,
                voice_activity_detector=(
                    self.voice_activity_detector
                    if self.use_vad
                    else lambda _waveform: [{'label': 'speech'}]), # type: ignore
                asr=self.asr,
                min_segment_size=self.min_segment_size,
                max_segment_size=self.max_segment_size,
            )
            texts.append(' '.join(text for _start, _end, text in segments)) # type: ignore
        return texts