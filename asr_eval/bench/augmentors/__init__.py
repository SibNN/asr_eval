from asr_eval import has_extras_package
from asr_eval.bench.augmentors._registry import (
    augmentors_registry,
    augmentors_instances,
    AudioAugmentor,
    get_augmentor,
    unload_augmentor,
)

if has_extras_package:
    import asr_eval_extras.audiobench.augmentors # type: ignore


__all__ = [
    'augmentors_registry',
    'AudioAugmentor',
    'augmentors_instances',
    'get_augmentor',
    'unload_augmentor',
]