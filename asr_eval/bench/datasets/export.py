import argparse
from collections.abc import Iterable
from typing import cast

import pandas as pd

from asr_eval.bench.datasets import get_dataset, AudioSample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        default='multivariant-v2',
        help='A registered dataset name to export annotations.',
    )
    parser.add_argument(
        'output',
        default='out.csv',
        help='Output CSV file.',
    )
    args = parser.parse_args()

    dataset = cast(Iterable[AudioSample], get_dataset(args.dataset_name))
    df = pd.DataFrame([
        {
            'dataset_name': args.dataset_name,
            'sample_id': sample['sample_id'],
            'text': sample['transcription'],
        }
        for sample in dataset
    ])
    df.to_csv(args.output, index=False)  # pyright: ignore[reportUnknownMemberType]

    print('Done!')