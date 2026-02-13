from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asr_eval.bench.datasets._registry import AudioSample
    


def remove_samples(
    sample: AudioSample, sample_ids_to_remove: set[int]
) -> bool:
    return sample['sample_id'] not in sample_ids_to_remove


def keep_samples(
    sample: AudioSample, sample_ids_to_keep: set[int]
) -> bool:
    return sample['sample_id'] in sample_ids_to_keep


def replace_transcriptions(
    sample: AudioSample, replacements: dict[int, str]
) -> dict[str, Any]:
    return {'transcription': replacements[sample['sample_id']]}


def add_empty_transcription_column(
    sample: dict[str, Any]
) -> dict[str, Any]:
    return {'transcription': ''}


def assign_sample_ids(
    sample: dict[str, Any], sample_idx: int
) -> dict[str, Any]:
    return {'sample_id': sample_idx}


def remove_samples_with_none_transcription(
    sample: dict[str, Any]
) -> bool:
    return sample['transcription'] is not None