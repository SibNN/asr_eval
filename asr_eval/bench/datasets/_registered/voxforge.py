import json
from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval import ROOT_DIR
from asr_eval.bench.datasets._registry import register_dataset, set_filter
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('voxforge-ru', splits=('train', 'test'))
def load_voxforge_ru(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'dangrebenkin/voxforge-ru-dataset',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )

@set_filter('voxforge-ru')
def filter_voxforge_ru(split: str = 'test') -> list[int]:
    data = (ROOT_DIR / '_data/duplicates/voxforge-ru.json').read_text()
    return json.loads(data).get(split, [])