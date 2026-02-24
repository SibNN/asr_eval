# Function find_audio_duplicates (defined in asr_eval/utils/deduplicate.py at lines 143-192)

def find_audio_duplicates(
    dataset: Dataset,
    window_size: int = 16_000,
    num_proc: int = 32,
) -> set[asr_eval.utils.deduplicate.Duplicate]:
    """Finds duplicates even with different normalization constant or
    different slicing. For example, if audio B is a copy of A, but
    sliced from 1 to 5 seconds, and multiplied by 2, will still detect
    it as a duplicate.

    It does the following:
    1. applies :code:`np.sign(np.diff(waveform)).astype(np.int8)` to
       each waveform
    2. in each waveform, finds all positions where ANCHOR is found
       (usually every ~0.1 sec)
    3. for each position P, extracts integer hash of
       :code:`waveform[P:P+window_size]`.
    4. also extracts integer hashes for the whole waveforms
    5. if equal hash is found for two different samples, adds them to
       duplicates set
    6. if this is the whole audio hash, sets :code:`mode='whole'`,
       otherwise :code:`mode='partial'`.
    """
    ...