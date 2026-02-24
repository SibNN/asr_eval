# Function set_filter (defined in asr_eval/bench/datasets/_registry.py at lines 169-183)

def set_filter(dataset_name: str):
    """Register a sample filter for the given registered dataset.

    The filter should accept split name and return a list of sample IDs
    to filter out. Is primarily used for deduplication. The
    :func:`~asr_eval.bench.datasets.get_dataset` function by default
    returns a filtered dataset if filter was set.
    """
    ...