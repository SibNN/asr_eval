import json
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, Features, Value # type: ignore

from asr_eval import ROOT_DIR
from asr_eval.bench.datasets._registry import (
    DATASETS_DIR, register_dataset, set_filter
)
from asr_eval.bench.datasets.mappers import assign_sample_ids



def load_openstt_subset(dir: Path) -> Dataset:
    samples: list[dict[str, Any]] = []
    for txt_path in sorted(dir.rglob('*.txt')):
        if (audio_path := txt_path.with_suffix('.wav')).exists():
            samples.append({
                'audio': str(audio_path),
                'transcription': txt_path.read_text().strip()
            })
    return Dataset.from_list( # type: ignore
        mapping=samples,
        features=Features({
            'audio': Audio(sampling_rate=16000),
            'transcription': Value('string'),
        })
    )
    
# OpenSTT datasets (those with human labeling):
# mkdir openstt && cd openstt
# wget https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_calls_2_val.tar.gz
# wget https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/buriy_audiobooks_2_val.tar.gz
# wget https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700_val.tar.gz
# tar -xvzf asr_calls_2_val.tar.gz && rm asr_calls_2_val.tar.gz
# tar -xvzf buriy_audiobooks_2_val.tar.gz && rm buriy_audiobooks_2_val.tar.gz
# tar -xvzf public_youtube700_val.tar.gz && rm public_youtube700_val.tar.gz

@register_dataset('openstt-asr-calls-2-val')
def load_openstt_asr_calls_2_val(split: str = 'test') -> Dataset:
    # openstt-asr-calls-2-val has only "val" split, called "test" in asr_eval
    return (
        load_openstt_subset(DATASETS_DIR / 'openstt/asr_calls_2_val')
        .map(assign_sample_ids, with_indices=True) # type: ignore
    )
    
@set_filter('openstt-asr-calls-2-val')
def filter_openstt_asr_calls_2_val(split: str = 'test') -> list[int]:
    data = (
        (ROOT_DIR / '_data/duplicates/openstt-asr-calls-2-val.json')
        .read_text()
    )
    return json.loads(data).get(split, [])

@register_dataset('openstt-buriy-audiobooks-2-val')
def load_openstt_buriy_audiobooks_2_val(split: str = 'test') -> Dataset:
    # openstt-buriy-audiobooks-2-val has only "val" split, called "test"
    # in asr_eval
    return (
        load_openstt_subset(DATASETS_DIR / 'openstt/buriy_audiobooks_2_val')
        .map(assign_sample_ids, with_indices=True) # type: ignore
    )

@register_dataset('openstt-public-youtube700-val')
def load_openstt_public_youtube700_val(split: str = 'test') -> Dataset:
    # openstt-public-youtube700-val has only "val" split, called "test"
    # in asr_eval
    return (
        load_openstt_subset(DATASETS_DIR / 'openstt/public_youtube700_val')
        .map(assign_sample_ids, with_indices=True) # type: ignore
    )