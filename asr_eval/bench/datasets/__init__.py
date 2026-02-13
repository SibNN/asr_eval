from asr_eval import has_datasets_package, has_extras_package
from asr_eval.bench.datasets._registry import (
    DATASETS_DIR,
    AudioData,
    AudioSample,
    DatasetInfo,
    datasets_registry,
    register_dataset,
    set_filter,
    get_dataset,
    get_filter,
    get_dataset_index,
    get_dataset_info,
    get_dataset_sample_by_id,
)
from asr_eval.bench.datasets.dataset_spec import DatasetSpec

if has_datasets_package:
    # registering datasets
    import asr_eval.bench.datasets._registered.audioset # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.fleurs # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.golos # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.multivariant # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.openstt # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.podlodka # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.rulibrispeech # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.sova # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.speech_massive # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.toloka # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.voxforge # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.yandex_commands # pyright: ignore[reportUnusedImport]
    import asr_eval.bench.datasets._registered.youtube_lectures # pyright: ignore[reportUnusedImport]
    if has_extras_package:
        import asr_eval_extras.audiobench.datasets # type: ignore
        

__all__ = [
    'AudioSample',
    'AudioData',
    'register_dataset',
    'datasets_registry',
    'DatasetInfo',
    'DATASETS_DIR',
    'DatasetSpec',
    'get_dataset',
    'get_dataset_index',
    'get_dataset_info',
    'get_dataset_sample_by_id',
    'set_filter',
    'get_filter',
]