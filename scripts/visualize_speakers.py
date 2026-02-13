from pathlib import Path
from asr_eval.bench.datasets import datasets_registry, get_dataset
from asr_eval.utils.deduplicate import visualize_speaker_embeddings


for dataset_name, dataset_info in datasets_registry.items():
    print(dataset_name)
    save_path = f'tmp/speaker_embeddings/{dataset_name}_embeddings.png'
    if not Path(save_path).exists():
        dataset_dict = {
            split_name: get_dataset(dataset_name, split_name) for split_name in dataset_info.splits
        }
        if min([len(split) for split in dataset_dict.values()]) > 50:
            visualize_speaker_embeddings(dataset_dict, show=False, save_path=save_path)