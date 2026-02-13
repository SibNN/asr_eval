from argparse import ArgumentParser
from collections.abc import Iterable
from typing import cast

import numpy as np

from asr_eval.bench.datasets import AudioSample, datasets_registry, get_dataset, get_filter


parser = ArgumentParser()
parser.add_argument('output', type=str, help='Output .rst file')
args = parser.parse_args()

header = '''\
.. list-table::
   :align: left
   :width: 80%
   :widths: 10 50
   :stub-columns: 1

   * - Name
     - {dataset_name}
   * - Splits\
'''

hist = '''\
   * - Lengths
     - .. raw:: html

          Audio lengths from {vmin:.1f} s to {vmax:.1f} s, typically from {q1:.1f} s to {q2:.1f} s
          <br><details><summary><a>See the length histogram for the test split</a></summary>
       .. image:: images/length_histograms/{dataset_name}_lengths.png
       .. raw:: html
        
          </details>\
'''

with open(args.output, 'w') as h:
    for dataset_name, info in datasets_registry.items():
        h.write(header.format(dataset_name=dataset_name) + '\n')
        h.flush()
        splits = list(info.splits)
        if 'test' in splits:
            splits.remove('test')
            splits = ['test'] + splits
        for split_idx, split_name in enumerate(splits):
            dataset = get_dataset(dataset_name, split_name, filter=True)
            total_len = sum([len(sample['audio']['array']) for sample in dataset]) / 16_000 # type: ignore
            filtered_out = get_filter(dataset_name, split_name)
            report = f'{split_name} - {len(dataset)} samples, {total_len / 3600:.2f} hours'
            if split_name == 'test':
                report = '**' + report + '**'
            h.write(
                ('       | ' if split_idx > 0 else '     - | ')
                + report
                + (f' (after removing {len(filtered_out)} duplicates)' if len(filtered_out) else '')
                + '\n'
            )
            h.flush()
        
        if 'test' in splits:
            dataset = get_dataset(dataset_name, 'test')
            lengths = [
                len(sample['audio']['array']) / 16_000
                for sample in cast(Iterable[AudioSample], dataset)
            ]
            q1, q2 = np.quantile(lengths, [0.2, 0.8])
            h.write(hist.format(
                dataset_name=dataset_name,
                vmin=min(lengths),
                vmax=max(lengths),
                q1=q1,
                q2=q2,
            ))
            h.flush()
            
        
        h.write('\n')
        h.flush()