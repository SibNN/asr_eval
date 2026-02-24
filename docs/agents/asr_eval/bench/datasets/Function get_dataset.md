# Function get_dataset (defined in asr_eval/bench/datasets/_registry.py at lines 197-257)

@functools.cache
def get_dataset(
    name: str,
    augmentor_name: str | None | typing.Literal['none'] = None,
    split: str = 'test',
    shuffle: bool = True,
    filter: bool = True,
) -> Dataset:
    """Instantiates a registered dataset.

    Args:
        name: A dataset name under which it was registered.
        augmentor_name: An augmentor name to apply, None by default
            (see :class:`~asr_eval.bench.augmentors.AudioAugmentor`).
        split: A split name, "test" by default.
        shuffle: Whether to perform :code:`shuffle(seed=0)`, True by
            default. The shuffling is used to ensure that the first N
            samples form a representative set. The sample IDs help to
            track the original indices, before shuffling or filtering.
        filter: Whether to filter out duplicate and malformed samples,
            if :class:`~asr_eval.bench.datasets.set_filter` was done
            for this dataset. True by default. This ensures that the
            datasets in asr_eval by default do not contain duplicate
            of malformed samples.
    """
    ...