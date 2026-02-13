import json
from pathlib import Path
from typing import Any, cast

from datasets import Audio, load_dataset, Dataset  # type: ignore
import requests

from asr_eval import ROOT_DIR
from asr_eval.bench.datasets._registry import (
    DATASETS_DIR, register_dataset, set_filter
)
from asr_eval.bench.datasets.mappers import assign_sample_ids


def _add_audio_column_to_toloka_VoxDIY_RusNews(
    sample: dict[str, Any], sample_idx: int
) -> dict[str, Any]:
    path = f'{DATASETS_DIR}/toloka_VoxDIY_RusNews/{sample_idx}.wav'
    return {'audio': {'path': path}}

def prepare_toloka_VoxDIY_RusNews(
    output_dir: str | Path = DATASETS_DIR / 'toloka_VoxDIY_RusNews'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset = load_dataset(
        'toloka/VoxDIY-RusNews', trust_remote_code=True, split='train'
    )
    for i, sample in enumerate(dataset): # type: ignore
        url = cast(str, sample['task']) # type: ignore
        ext = Path(url).suffix
        assert ext == '.wav'
        output_path = output_dir / f'{i}{ext}'
        if output_path.exists():
            continue
        response = requests.get(url)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f'{i}: done!')
        else:
            print(
                f'{i}: Failed to download file.'
                f' Status code: {response.status_code}'
            )

@register_dataset('toloka-voxdiy-rusnews')
def load_toloka_VoxDIY_RusNews(split: str = 'test') -> Dataset:
    # toloka/VoxDIY-RusNews has a single split, called "test" in asr_eval
    # and "train" on HF
    dataset = cast(Dataset, load_dataset(
        'toloka/VoxDIY-RusNews', trust_remote_code=True, split='train'
    )) # pyright: ignore[reportUnnecessaryCast]
    return (
        dataset
        .map(_add_audio_column_to_toloka_VoxDIY_RusNews, with_indices=True) # type: ignore
        .cast_column('audio', Audio(sampling_rate=16_000, decode=True))
        .rename_column('gt', 'transcription')
        .map(assign_sample_ids, with_indices=True)
    )

@set_filter('toloka-voxdiy-rusnews')
def filter_toloka_VoxDIY_RusNews(split: str = 'test') -> list[int]:
    data = (
        (ROOT_DIR / '_data/duplicates/toloka-voxdiy-rusnews.json').read_text()
    )
    return json.loads(data).get(split, [])