import typing
import sys
from pathlib import Path
import math
import datetime

import torch
from datasets import load_from_disk, Audio, Dataset # type: ignore
import gigaam
from gigaam.model import GigaAMASR
import numpy as np
import numpy.typing as npt

sys.path.append("/userspace/soa/asr-eval")
from asr_eval.models.gigaam import transcribe_with_gigaam_ctc


model = typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
batch_size = 256

sova_dir = Path('/storage2/soa/sova')
output_dir = Path('/storage2/soa/sova_gigaam2')

for sova_part_dir in sorted(sova_dir.glob('*-of-00500')):
    print(f'Processing {sova_part_dir}')
    
    if (output_part_dir := output_dir / sova_part_dir.name).exists():
        print('Already exists')
        continue
    
    sova_part: Dataset = load_from_disk(sova_part_dir).cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore

    log_probs: list[npt.NDArray[np.float32 | np.float16]] = []
    texts: list[str] = []
    
    for batch_idx, batch in enumerate(sova_part.iter(batch_size=batch_size)): # type: ignore
        outputs = transcribe_with_gigaam_ctc(model, [x['array'] for x in batch['audio']]) # type: ignore
        log_probs += [x.log_probs for x in outputs]
        texts += [x.text for x in outputs]
        print(
            f'[{datetime.datetime.now()}]'
            f' Done {batch_idx}/{math.ceil(len(sova_part) / batch_size)}'
            f' Max memory allocated {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GB',
            flush=True
        )
        # break
    
    Dataset.from_dict({
        'gigaam2_log_probs': log_probs,
        'gigaam2_texts': texts,
    }).save_to_disk(output_part_dir) # type: ignore
    # break