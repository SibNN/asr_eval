import argparse
import numpy as np
import matplotlib.pyplot as plt

from asr_eval.bench.loader import PredictionLoader
from asr_eval.bench.evaluator import get_dataset_data
from asr_eval.utils.storage import make_storage


parser = argparse.ArgumentParser()
parser.add_argument('--base', default='openai/whisper-medium')
parser.add_argument('--dataset-multi', default='sova-rudevices-multivariant')
parser.add_argument('--dataset-single', default='sova-rudevices-single-variant')
parser.add_argument('--storage', default='tmp/tuning_evals.db')
parser.add_argument('--cache', default='tmp/tuning_evals_cache.db')
parser.add_argument('--n_steps', type=int, default=40)
parser.add_argument('--output', default='tmp/tuning_plot.png')
args = parser.parse_args()

loader = PredictionLoader(
    storage=make_storage(args.storage),
    cache=make_storage(args.cache),
    dataset_specs=[
        f'{args.dataset_single}:p=default,ru-norm,ru-norm-lite:n=all!',
        f'{args.dataset_multi}:n=all!',
    ],
)

plt.figure(figsize=(5, 4)) # type: ignore

for name_idx, (name, config) in enumerate({
    'orig': {
        'dataset_name': args.dataset_single,
        'parser_name': 'default',
    },
    'normalized': {
        'dataset_name': args.dataset_single,
        'parser_name': 'ru-norm',
    },
    'normalized-lite': {
        'dataset_name': args.dataset_single,
        'parser_name': 'ru-norm-lite',
    },
    'multivariant': {
        'dataset_name': args.dataset_multi,
        'parser_name': 'default',
    },
}.items()):
    print(f'{name}...')
    multiple_aligments = loader.get_multiple_alignments(
        dataset_name=config['dataset_name'],
        parser_name=config['parser_name'],
        pipeline_patterns=[f'{args.base}_step_*'],
    )

    dataset_metric = get_dataset_data(
        multiple_aligments,
        max_consecutive_insertions=4,
        exclude_samples_with_digits=False,
        max_samples_to_render=0,
    ).dataset_metric
    
    wer_distributions = [metric.wer for metric in dataset_metric.values()]
    steps = np.array([int(name.split('_')[-1]) for name in dataset_metric.keys()])

    wer_main = np.array([d.main_value for d in wer_distributions])
    wer_lower = np.array([d.quantiles([0.1])[0] for d in wer_distributions])
    wer_upper = np.array([d.quantiles([0.9])[0] for d in wer_distributions])

    order = np.argsort(steps)
    steps = steps[order][:args.n_steps]
    wer_main = wer_main[order][:args.n_steps]
    wer_lower = wer_lower[order][:args.n_steps]
    wer_upper = wer_upper[order][:args.n_steps]
    
    plt.plot( # type: ignore
        steps,
        wer_main,
        label=name,
        color=f'C{name_idx}',
    )
    plt.fill_between( # type: ignore
        steps,
        wer_lower,
        wer_upper,
        color=f'C{name_idx}',
        alpha=0.1,
    )
    plt.xlabel('Step') # type: ignore
    plt.ylabel('Word Error Rate') # type: ignore

plt.legend() # type: ignore
plt.savefig(args.output) # type: ignore