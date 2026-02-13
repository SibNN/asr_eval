"""A registry for pipelines."""

from asr_eval import has_extras_package
from asr_eval.bench.pipelines._registry import (
    TranscriberPipeline,
    pipelines_registry,
    get_pipeline,
    get_pipeline_index,
)

# registering pipelines
import asr_eval.bench.pipelines._registered.api # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.audio_llm # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.gigaam # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.nemo # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.pisets # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.t_one # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.vosk # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.whisper # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.wav2vec2 # pyright: ignore[reportUnusedImport]
import asr_eval.bench.pipelines._registered.speechbrain # pyright: ignore[reportUnusedImport]
if has_extras_package:
    import asr_eval_extras.audiobench.pipelines # type:ignore


__all__ = [
    'pipelines_registry',
    'TranscriberPipeline',
    'get_pipeline',
    'get_pipeline_index',
]