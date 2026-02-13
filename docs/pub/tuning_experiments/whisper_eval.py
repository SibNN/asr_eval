from itertools import count
from pathlib import Path
from typing import Sequence, cast
import argparse

from datasets import load_from_disk # type: ignore

from asr_eval.bench.datasets import AudioSample, get_dataset
from asr_eval.utils.storage import make_storage
from asr_eval.models.whisper_wrapper import WhisperLongformWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--base', default='openai/whisper-medium')
parser.add_argument('--tuned_pattern', default='tmp/tuning/whisper-medium-rudevices/checkpoint-{step}')
parser.add_argument('--eval_period', type=int, default=100)
parser.add_argument('--dataset-multi', default='sova-rudevices-multivariant')
parser.add_argument('--dataset-single', default='sova-rudevices-single-variant')
parser.add_argument('--storage', default='tmp/tuning_evals.db')
args = parser.parse_args()

storage = make_storage(args.storage)

try:
    dataset = get_dataset(args.dataset_multi)
    dataset.save_to_disk(f'tmp/__cached_{args.dataset_multi}') # type: ignore
except ValueError:
    # a fallback for weird cluster behaviour
    # (Couldn't find cache for sova_rudevices for config .....)
    dataset = load_from_disk(f'tmp/__cached_{args.dataset_multi}')
    
whsiper_lang = 'ru'

for step in count(step=args.eval_period):
    ckpt = args.tuned_pattern.format(step=step) if step > 0 else args.base

    if step > 0 and not Path(ckpt).exists():
        print(f'Stop iterating on {ckpt} (not found)')
        break
    print(f'Loading {ckpt}')
    
    model = WhisperLongformWrapper(ckpt, preproc_name=args.base, lang=whsiper_lang)
    for sample_idx, sample in enumerate(cast(Sequence[AudioSample], dataset)):
        print(f'Sample {sample_idx}/{len(dataset)} checkpoint {ckpt}')
        key = dict(
            pipeline_name=f'{args.base}_step_{step:05d}',
            dataset_name=args.dataset_multi,
            augmentor='none',
            sample_id=sample['sample_id'],
            artifact_type='text',
        )
        has_row = storage.has_row(**key)
        if not has_row:
            text = model.transcribe(sample['audio']['array'])
            print(f'Prediction: {text}', flush=True)
            storage.add_row(text, overwrite=False, **key)
    
for row in storage.list_all().iter_rows(named=True):
    # since dataset-multi and dataset-single have different annotations, but
    # the same audios, we can use the same model prediction for both versions
    value = storage.get_row(**row)
    row['dataset_name'] = args.dataset_single
    storage.add_row(value, **row)