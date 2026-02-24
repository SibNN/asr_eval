# Function find_audio_duplicates_for_multiple_splits (defined in asr_eval/utils/deduplicate.py at lines 193-271)

def find_audio_duplicates_for_multiple_splits(
    splits: dict[str, Dataset] | DatasetDict,
    splits_order: typing.Sequence[str],
    window_size: int = 16_000,
    num_proc: int = 32,
) -> pd.DataFrame:
    """A generalization of
    :func:`~asr_eval.utils.deduplicate.find_audio_duplicates` that is
    applicable to a datset with multiple splits.

    Forms a dataframe with columns:
    - dup_split - a split of a duplicated sample
    - dup_idx - a positional index of a duplicated sample
    - orig_split - a split of the original sample
    - orig_idx - a positional index of the original sample
    - mode - if duplicate is "whole" or "partial"

    If two duplicated samples are found in different splits, their split
    indices in :code:`splits_order` are compared: the smaller split
    index is considered original, and the larger is considered
    duplicated. So, if you dataset has "train", "val" and "test"
    splits, specify :code:`splits_order=['train', 'val', 'test']`. This
    ensures that if a sample is found in train and test splits, it will
    be considered duplicate (to remove later) in the test split.
    """
    ...