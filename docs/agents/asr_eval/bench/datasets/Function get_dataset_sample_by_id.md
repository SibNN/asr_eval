# Function get_dataset_sample_by_id (defined in asr_eval/bench/datasets/_registry.py at lines 289-320)

def get_dataset_sample_by_id(
    dataset_name: str,
    split: str,
    sample_id: int,
    augmentor_name: str | None = None,
) -> asr_eval.bench.datasets._registry.AudioSample:
    """An utility to simply retrieve the required sample ID for the
    given dataset. Internally instantiates a dataset if not instantiated
    yet.
    """
    ...