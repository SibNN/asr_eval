from pathlib import Path
from typing import cast
from asr_eval.bench.datasets import datasets_registry, get_dataset
from asr_eval.utils.deduplicate import find_audio_duplicates_for_multiple_splits
from asr_eval.utils.serializing import save_to_json


for dataset_name, dataset_info in datasets_registry.items():
    print(dataset_name)
    output_path = f'tmp/duplicates/{dataset_name}.json'
    if Path(output_path).exists():
        continue
    dataset_dict = {
        split_name: get_dataset(dataset_name, split_name) for split_name in dataset_info.splits
    }
    duplicates_df = find_audio_duplicates_for_multiple_splits(
        dataset_dict, splits_order=dataset_info.splits #, num_proc=1,
    )
    if len(duplicates_df):
        print(f'found {len(duplicates_df)} duplicates')
        duplicates_json = {
            dup_split: cast(list[int], group['dup_idx'].tolist())
            for dup_split, group in duplicates_df.groupby('dup_split') # type: ignore
        }
        save_to_json(duplicates_json, output_path)