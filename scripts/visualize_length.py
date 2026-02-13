from collections.abc import Iterable
from pathlib import Path
from typing import cast


from asr_eval.bench.datasets import AudioSample, datasets_registry, get_dataset
from asr_eval.utils.plots import plot_length_histogram

for dataset_name, dataset_info in datasets_registry.items():
    print(dataset_name)
    save_path = f'tmp/length_histograms/{dataset_name}_lengths.png'
    if not Path(save_path).exists():
        dataset = get_dataset(dataset_name, 'test')
        lengths = [
            len(sample['audio']['array']) / 16_000
            for sample in cast(Iterable[AudioSample], dataset)
        ]
        plot_length_histogram(lengths, show=False, save_path=save_path)