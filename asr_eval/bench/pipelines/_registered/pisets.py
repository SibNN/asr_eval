from __future__ import annotations

from asr_eval import CACHE_DIR
from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.legacy_pisets_wrapper import LegacyPisetsWrapper


PISETS_DIR = CACHE_DIR / 'pisets_legacy'


class _(TranscriberPipeline, register_as='pisets-legacy'):
    def init(self):
        return LegacyPisetsWrapper(
            repo_dir=str(PISETS_DIR)
        )


class _(TranscriberPipeline, register_as='pisets-legacy-turbo'):
    def init(self):
        return LegacyPisetsWrapper(
            repo_dir=str(PISETS_DIR),
            whisper_ckpt='bond005/whisper-podlodka-turbo',
        )