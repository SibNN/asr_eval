from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, override

from ..models.base.longform import ContextualLongformVAD, LongformCTC, LongformVAD
from ..models.flamingo_wrapper import FlamingoWrapper
from ..models.gemma_wrapper import Gemma3nWrapper
from ..models.gigaam_wrapper import GigaAMShortformCTC, GigaAMShortformRNNT
from ..models.legacy_pisets_wrapper import LegacyPisetsWrapper
from ..models.pisets_wrapper import PisetsWrapper
from ..models.pyannote_segmenter import PyannoteSegmenter
from ..models.qwen2_audio_wrapper import Qwen2AudioWrapper
from ..models.t_one_wrapper import TOneWrapper
from ..models.vosk54_wrapper import VoskV54
from ..models.voxtral_wrapper import VoxtralWrapper
from ..models.whisper_wrapper import WhisperLongformWrapper
from ..models.yandex_speechkit_wrapper import YandexSpeechKitWrapper
from ..models.base.interfaces import TimedTranscriber, Transcriber
from ..datasets.datasets import AudioSample
from ..utils.serializing import save_to_json


pipelines: dict[str, type[Pipeline]] = {}


class Pipeline(ABC):
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
        super().__init_subclass__(**kwargs)
        if register_as:
            assert register_as not in pipelines
            pipelines[register_as] = cls
            cls.name = register_as


class TranscriberPipeline(Pipeline):
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
        output = self.transcriber.transcribe(sample['audio']['array'])
        save_to_json(
            {'output': output, 'type': 'transcription'},
            sample_dir / 'transcription.json',
            indent=0
        )


class TimedTranscriberPipeline(Pipeline):
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
        output = self.transcriber.timed_transcribe(sample['audio']['array'])
        save_to_json(
            {'output': output, 'type': 'timed_transcription'},
            sample_dir / 'transcription.json',
            indent=0
        )
        

# TODO better check language for each transcriber
# TODO check if we need VAD for Vosk54


class _(TranscriberPipeline, register_as='whisper-tiny'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-tiny')


class _(TranscriberPipeline, register_as='whisper-large-v3'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-large-v3')


class _(TranscriberPipeline, register_as='gigaam-ctc'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC())


class _(TimedTranscriberPipeline, register_as='gigaam-ctc-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformCTC(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gigaam-rnnt-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformRNNT(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='flamingo-ru-vad'):
    def init(self):
        return LongformVAD(FlamingoWrapper(lang='ru'), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gemma3n-ru-vad'):
    def init(self):
        return LongformVAD(Gemma3nWrapper(lang='ru'), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='gemma3n-ru-vad-contextual'):
    def init(self):
        return ContextualLongformVAD(
            Gemma3nWrapper(lang='ru'), PyannoteSegmenter(), max_history_words=100
        )


class _(TimedTranscriberPipeline, register_as='pisets-legacy'):
    def init(self):
        return LegacyPisetsWrapper(repo_dir='tmp/pisets')


class _(TimedTranscriberPipeline, register_as='pisets-ru-whisper-large-v3'):
    def init(self):
        return PisetsWrapper(
            language='ru', recognizer='openai/whisper-large-v3', diarization=None
        )


class _(TimedTranscriberPipeline, register_as='qwen2-audio-vad'):
    def init(self):
        return LongformVAD(Qwen2AudioWrapper(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='t-one-vad'):
    def init(self):
        return LongformVAD(TOneWrapper(), PyannoteSegmenter())


class _(TimedTranscriberPipeline, register_as='vosk-0.54-vad'):
    def init(self):
        return LongformVAD(VoskV54(), PyannoteSegmenter())


class _(TranscriberPipeline, register_as='voxtral-3B'):
    def init(self):
        return VoxtralWrapper('mistralai/Voxtral-Mini-3B-2507', language='ru')


class _(TimedTranscriberPipeline, register_as='yandex-speechkit'):
    def init(self):
        return YandexSpeechKitWrapper(
            api_key=os.environ['YANDEX_API_KEY'], language='ru-RU', normalize=False
        )