# Function _extract_hashes_from_dataset (defined in asr_eval/utils/deduplicate.py at lines 100-122)

def _extract_hashes_from_dataset(
    dataset: Dataset,
    window_size: int = 16_000,
    num_proc: int = 32,
) -> dict[int, list[int]]:
    """Using :func:`~asr_eval.utils.deduplicate._extract_hashes`
    extracts hashes from the audio dataset. If a hash is found in two
    different samples, they are considered duplicate.

    Returns a mapping from a hash to a list of sample indices where it
    is found.
    """
    ...