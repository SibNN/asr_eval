from collections.abc import Iterable
from pathlib import Path
import pickle
from typing import cast

from tqdm.auto import tqdm

from asr_eval.align.parsing import parse_multivariant_string
from asr_eval.align.timings import fill_word_timings_inplace
from asr_eval.bench.datasets import AudioSample, get_dataset
from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC


# tmp/venv_gigaam/bin/python timings.py


model = GigaAMShortformCTC()
    
dataset_name = 'common-voice-17.0'
max_samples = 500

dataset = get_dataset(dataset_name)
if len(dataset) > max_samples:
    dataset = dataset.take(max_samples)

dataset = cast(Iterable[AudioSample], dataset)

for sample_idx, sample in enumerate(tqdm(dataset)):
    save_path = Path(f'tmp/timings/{dataset_name}/{sample_idx}.pkl')
    if not save_path.exists():
        transcription = parse_multivariant_string(sample['transcription'])
        fill_word_timings_inplace(model, sample['audio']['array'], transcription, verbose=True)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_path.write_bytes(pickle.dumps(transcription))