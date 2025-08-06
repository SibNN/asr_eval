from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, override

from ..models.base.longform import ContextualLongformVAD, LongformCTC, LongformVAD
from ..models.base.interfaces import TimedTranscriber, Transcriber
from .datasets import AudioSample
from ..utils.serializing import save_to_json


pipelines_registry: dict[str, type[Pipeline]] = {}


class Pipeline(ABC):
    FILENAME: str = ''
    
    @abstractmethod
    def run_on_dataset_sample(
        self,
        dataset_name: str,
        dataset_idx: int,
        sample: AudioSample,
        root_dir: Path,
        sample_dir: Path,
    ): ...
    
    def __init_subclass__(cls, register_as: str | None = None, **kwargs: Any):
        global pipelines_registry
        super().__init_subclass__(**kwargs)
        if register_as:
            assert register_as not in pipelines_registry
            pipelines_registry[register_as] = cls
            cls.name = register_as


class TranscriberPipeline(Pipeline):
    FILENAME: str = 'transcription.json'
    
    def __init__(self):
        self.transcriber = self.init()
    
    @abstractmethod
    def init(self) -> Transcriber: ...
    
    @override
    def run_on_dataset_sample(
        self,
        dataset_name: str,
        dataset_idx: int,
        sample: AudioSample,
        root_dir: Path,
        sample_dir: Path,
    ):
        output_path = sample_dir / self.FILENAME
        if output_path.exists():
            return
        output = self.transcriber.transcribe(sample['audio']['array'])
        save_to_json({'output': output, 'type': 'transcription'}, output_path, indent=0)


class TimedTranscriberPipeline(Pipeline):
    FILENAME: str = 'transcription.json'
    
    def __init__(self):
        self.transcriber = self.init()
    
    @abstractmethod
    def init(self) -> TimedTranscriber: ...
    
    @override
    def run_on_dataset_sample(
        self,
        dataset_name: str,
        dataset_idx: int,
        sample: AudioSample,
        root_dir: Path,
        sample_dir: Path,
    ):
        output_path = sample_dir / self.FILENAME
        if output_path.exists():
            return
        output = self.transcriber.timed_transcribe(sample['audio']['array'])
        save_to_json({'output': output, 'type': 'timed_transcription'}, output_path, indent=0)
        

# TODO better check language for each transcriber
# TODO check if we need VAD for Vosk54


class _(TranscriberPipeline, register_as='whisper-tiny'):
    def init(self):
        from ..models.whisper_wrapper import WhisperLongformWrapper
        return WhisperLongformWrapper('openai/whisper-tiny')

class _(TranscriberPipeline, register_as='whisper-small'):
    def init(self):
        from ..models.whisper_wrapper import WhisperLongformWrapper
        return WhisperLongformWrapper('openai/whisper-small')


class _(TranscriberPipeline, register_as='whisper-large-v3'):
    def init(self):
        from ..models.whisper_wrapper import WhisperLongformWrapper
        return WhisperLongformWrapper('openai/whisper-large-v3')


class _(TranscriberPipeline, register_as='gigaam-ctc'):
    def init(self):
        from ..models.gigaam_wrapper import GigaAMShortformCTC
        return LongformCTC(GigaAMShortformCTC())


class _(TimedTranscriberPipeline, register_as='gigaam-ctc-vad'):
    def init(self):
        from ..models.gigaam_wrapper import GigaAMShortformCTC
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(GigaAMShortformCTC(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gigaam-rnnt-vad'):
    def init(self):
        from ..models.gigaam_wrapper import GigaAMShortformRNNT
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(GigaAMShortformRNNT(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='flamingo-ru-vad'):
    def init(self):
        from ..models.flamingo_wrapper import FlamingoWrapper
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(FlamingoWrapper(lang='ru'), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gemma3n-ru-vad'):
    def init(self):
        from ..models.gemma_wrapper import Gemma3nWrapper
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(Gemma3nWrapper(lang='ru'), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gemma3n-ru-vad-contextual'):
    def init(self):
        from ..models.gemma_wrapper import Gemma3nWrapper
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return ContextualLongformVAD(
            Gemma3nWrapper(lang='ru'), PyannoteSegmenter(), max_history_words=100
        )


class _(TimedTranscriberPipeline, register_as='pisets-legacy'):
    def init(self):
        from ..models.legacy_pisets_wrapper import LegacyPisetsWrapper
        return LegacyPisetsWrapper(repo_dir='tmp/pisets')


class _(TimedTranscriberPipeline, register_as='pisets-ru-whisper-large-v3'):
    def init(self):
        from ..models.pisets_wrapper import PisetsWrapper
        return PisetsWrapper(
            language='ru', recognizer='openai/whisper-large-v3', diarization=None
        )


class _(TimedTranscriberPipeline, register_as='qwen2-audio-vad'):
    def init(self):
        from ..models.qwen2_audio_wrapper import Qwen2AudioWrapper
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(Qwen2AudioWrapper(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='t-one-vad'):
    def init(self):
        from ..models.t_one_wrapper import TOneWrapper
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(TOneWrapper(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='vosk-0.54-vad'):
    def init(self):
        from ..models.vosk54_wrapper import VoskV54
        from ..models.pyannote_segmenter import PyannoteSegmenter
        return LongformVAD(VoskV54(), PyannoteSegmenter())


class _(TranscriberPipeline, register_as='voxtral-3B'):
    def init(self):
        from ..models.voxtral_wrapper import VoxtralWrapper
        return VoxtralWrapper('mistralai/Voxtral-Mini-3B-2507', language='ru')


class _(TimedTranscriberPipeline, register_as='yandex-speechkit'):
    def init(self):
        from ..models.yandex_speechkit_wrapper import YandexSpeechKitWrapper
        return YandexSpeechKitWrapper(
            api_key=os.environ['YANDEX_API_KEY'], language='ru-RU', normalize=False
        )