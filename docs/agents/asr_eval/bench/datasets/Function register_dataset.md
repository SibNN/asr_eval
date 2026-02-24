# Function register_dataset (defined in asr_eval/bench/datasets/_registry.py at lines 123-167)

def register_dataset(
    name: str,
    splits: tuple[str, ...] = ('test',),
    unlabeled: bool = False,
):
    """Register a new dataset in asr_eval. The dataset will be available
    under the registered name in
    :func:`~asr_eval.bench.datasets.get_dataset`.

    Args:
        name: A unique name for the dataset.
        splits: A list of available splits. All the datasets should at
            least have "test" split available, because asr_eval is for
            testing purposes. If a dataset has a "train" split only,
            consider registering it as "test" if you want to test on it.
            Datasets can have other splits registered with any names,
            primarily to check for train-test overlap.
        unlabeled: If the dataset is unlabeled. Experimental feature.

    See many examples in `asr_eval.bench.datasets._registered` package.

    Example:
        >>> from datasets import Audio, load_dataset, Dataset
        >>> from asr_eval.bench.datasets import register_dataset, get_dataset
        >>> from asr_eval.bench.datasets.mappers import assign_sample_ids
        >>> @register_dataset('podlodka-new', splits=('train', 'test'))
        >>> def load_podlodka(split: str = 'test') -> Dataset:
        ...     return (
        ...         load_dataset('bond005/podlodka_speech', split=split)
        ...         .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ...         .map(assign_sample_ids, with_indices=True)
        ...     )
        >>> dataset = get_dataset('podlodka-new')
    """
    ...