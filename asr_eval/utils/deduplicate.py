from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict # type: ignore

from asr_eval.bench.datasets import AudioSample
from asr_eval.utils.misc import locate_subarray_in_array
from asr_eval.utils.types import FLOATS, INTS


__all__ = [
    'find_audio_duplicates',
    'find_audio_duplicates_for_multiple_splits',
    'Duplicate',
    '_extract_hashes_from_dataset',
    'visualize_speaker_embeddings',
]


def _preproc_waveform(waveform: FLOATS) -> INTS:
    """Preprocesses float32 waveform for duplicate search purpose.
    
    1. Taking sign accounts for possible different normalization.
    2. Taking diff addresses cases when values are greater than
       zero for hundreds of frames: taking only sign gives an array
       of ones, which may give false positive results when searching
       for duplicates.
    """
    
    return np.sign(np.diff(waveform)).astype(np.int8)

def _hash_waveform(waveform_processed: INTS) -> int:
    """A fingerprint for a slice of waveform after
    :func:`~asr_eval.utils.deduplicate._preproc_waveform`, for duplicate
    search purpose.
    
    We take abs because we later use sign to store additional
    information.
    """
    
    return abs(hash(waveform_processed.tobytes()))

_ANCHOR = np.array([-1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1])
"""An anchor that we search in a waveform. Acts as a stage for duplicate
search purpose.
"""

def _find_anchors(waveform_processed: INTS) -> list[int]:
    """Serches for all positions where waveform part matches
    :data:`~asr_eval.utils.deduplicate._ANCHOR`. The :code:`ANCHOR`
    value was chosen randomly. Experimentally, this happens nearly every
    0.1 second in each audio.
    """
    
    if len(waveform_processed) < len(_ANCHOR):
        return []
    return locate_subarray_in_array(waveform_processed, _ANCHOR)

def _extract_hashes(
    waveform: FLOATS,
    window_size: int = 16_000,
) -> list[int]:
    """Extrach hashes: (with positive sign) for the whole audio, and
    (with negative sign) for each of the audio parts found with
    :code:`find_anchors`.
    """
    
    waveform_processed = _preproc_waveform(waveform)
    total_hash = _hash_waveform(waveform_processed)
    anchor_positions = _find_anchors(waveform_processed)
    ahchor_hashes = [
        -_hash_waveform(waveform_processed[pos:pos + window_size])
        for pos in anchor_positions
        if pos <= len(waveform_processed) - window_size
    ]
    return [total_hash] + ahchor_hashes

def _extract_hashes_mapper(
    sample: AudioSample,
    window_size: int = 16_000,
) -> dict[str, list[int]]:
    """A Dataset.map() wrapper for
    :func:`~asr_eval.utils.deduplicate._extract_hashes`.
    """
    
    waveform = sample['audio']['array']
    return {'hashes': _extract_hashes(waveform, window_size=window_size)}

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
    results: dict[int, list[int]] = defaultdict(list)
    mapped = dataset.map( # type: ignore
        partial(_extract_hashes_mapper, window_size=window_size),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    for sample_idx, hashes in enumerate(mapped): # type: ignore
        for _hash in hashes['hashes']: # type: ignore
            results[_hash].append(sample_idx)
    return results

@dataclass
class Duplicate:
    """An information about found duplicate."""
    
    mode: Literal['whole', 'partial']
    """If "partial" this is a duplicate with different slicing. For
    example, if sample #1 has a length of 10 seconds, and sample #0 is
    a slice of sample #1 from 3 to 7 seconds, then they both form a
    :code:`Duplicate(mode='partial', sample_idxs=[0, 1])`.
    """
    
    sample_idxs: list[int]
    """A list of sample indices that are considered duplicates."""
    
    def __hash__(self) -> int:
        return hash((
            self.mode,
            tuple(self.sample_idxs),
        ))

def find_audio_duplicates(
    dataset: Dataset,
    window_size: int = 16_000,
    num_proc: int = 32,
) -> set[Duplicate]:
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
    
    duplicates: set[Duplicate] = set()
    hashes = _extract_hashes_from_dataset(
        dataset, window_size=window_size, num_proc=num_proc
    )
    
    for _hash, sample_idxs in hashes.items():
        if len(sample_idxs) > 1:
            duplicates.add(Duplicate(
                mode='partial' if _hash < 0 else 'whole',
                sample_idxs=sample_idxs,
            ))
    
    whole_duplicates = {
        d for d in duplicates
        if d.mode == 'whole'
    }
    
    # excluding partial duplicates which are the same as whole duplicates
    partial_duplicates = {
        d for d in duplicates
        if d.mode == 'partial'
        and replace(d, mode='whole') not in whole_duplicates
    }
        
    return whole_duplicates | partial_duplicates

def find_audio_duplicates_for_multiple_splits(
    splits: dict[str, Dataset] | DatasetDict,
    splits_order: Sequence[str],
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
    
    from datasets import concatenate_datasets
    splits = cast(dict[str, Dataset], dict(splits)) # type: ignore
    
    datasets = [splits[split_name] for split_name in splits_order]
    split_sizes = [len(dataset) for dataset in datasets]
    
    duplicates = list(find_audio_duplicates(
        concatenate_datasets(datasets),
        window_size=window_size,
        num_proc=num_proc,
    ))
    
    dup_to_orig: dict[int, dict[str, Any]] = {}
    for duplicate in duplicates:
        sample_idxs = sorted(duplicate.sample_idxs)
        orig, to_remove = sample_idxs[0], sample_idxs[1:]
        for idx in to_remove:
            existing_orig = dup_to_orig.get(idx, None)
            if (
                existing_orig is None
                or (
                    existing_orig['mode'] == 'partial'
                    and duplicate.mode == 'whole'
                )
                or (
                    existing_orig['mode'] == duplicate.mode
                    and existing_orig['orig'] > orig
                )
            ):
                dup_to_orig[idx] = {
                    'orig': orig, 'mode': duplicate.mode
                }

    split_name_mask: list[str] = []
    split_idx_mask: list[int] = []
    for split_name, split_size in zip(splits, split_sizes):
        split_name_mask += [split_name] * split_size
        split_idx_mask += list(range(split_size))

    df_rows: list[dict[str, Any]] = []
    for dup_idx, orig_info in dup_to_orig.items():
        orig_idx = cast(int, orig_info['orig'])
        df_rows.append({
            'dup_split': split_name_mask[dup_idx],
            'dup_idx': split_idx_mask[dup_idx],
            'orig_split': split_name_mask[orig_idx],
            'orig_idx': split_idx_mask[orig_idx],
            'mode': orig_info['mode'],
        })

    df = pd.DataFrame(df_rows)
    return df


def visualize_speaker_embeddings(
    splits: dict[str, Dataset] | DatasetDict,
    split_colors: dict[str, str] | None = None,
    max_samples_per_split: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[FLOATS, FLOATS]:
    """Performs speaker embedding analysis via UMAP projection into a 2D
    plot. Draws the plot and saves to the :code:`save_path`. Returns
    speaker embeddings, both original and after UMAP.
    
    Requires :code:`pip install torch umap-learn pyannote.audio`
    """
    
    splits = cast(dict[str, Dataset], dict(splits)) # type: ignore
    
    import torch
    import umap  # type: ignore
    from pyannote.audio import Inference  # type: ignore
    from pyannote.audio import Model  # type: ignore
    
    if max_samples_per_split is not None:
        splits = {
            split_name: (
                split.take(max_samples_per_split)
                if len(split) > max_samples_per_split
                else split
            )
            for split_name, split in splits.items()
        }
    
    model = Model.from_pretrained('pyannote/embedding') # type: ignore
    assert model is not None
    inference = Inference(model, window='whole')
    
    embeddings_list: list[FLOATS] = []
    split_sizes: list[int] = []
    
    for split in splits.values():
        i = 0
        for sample in tqdm(cast(Iterable[AudioSample], split)):
            waveform = sample['audio']['array']
            if len(waveform) > 8_000:
                embedding = cast(FLOATS, inference({
                    'waveform': torch.tensor(waveform, dtype=torch.float32) \
                        .unsqueeze(0),
                    'sample_rate': 16000,
                }))
                embeddings_list.append(embedding)
                i += 1
        split_sizes.append(i)
    
    split_names = list(splits.keys()) # type: ignore

    split_name_mask: list[str] = []
    split_idx_mask: list[int] = []
    for split_name, split_size in zip(split_names, split_sizes):
        split_name_mask += [split_name] * split_size
        split_idx_mask += list(range(split_size))
    
    X = np.stack(embeddings_list)

    # Standardize (important for PCA)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = cast(FLOATS, scaler.fit_transform(X)) # type: ignore
    
    pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
    X_pca = pca.fit_transform(X_std) # type: ignore
    
    umap_model = umap.UMAP( # type: ignore
        n_neighbors=50,
        min_dist=0.15,
        n_components=2,
        metric='cosine',
        random_state=0,
        low_memory=False
    )
    X_umap = cast(FLOATS, umap_model.fit_transform(X_pca)) # type: ignore
    
    colors = split_colors or {
        'train': 'lightblue', 'validation': 'lightblue', 'test': 'r'
    }
    color_mask = [
        (colors[x] if x in colors else 'black') for x in split_name_mask
    ]

    _fig, ax = plt.subplots(figsize=(8, 8)) # type: ignore
    ax.scatter( # type: ignore
        X_umap[:, 0],
        X_umap[:, 1],
        s=10 if len(color_mask) < 1000 else 1,
        alpha=0.3,
        c=color_mask,
        rasterized=True,     # speeds up vector backends (PDF/SVG)
    )
    ax.set_xlabel("UMAP-1") # type: ignore
    ax.set_ylabel("UMAP-2") # type: ignore
    ax.set_title("Speaker embeddings") # type: ignore
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(save_path)) # type: ignore
    if show:
        plt.show() # type: ignore
    plt.close()
    
    return X, X_umap