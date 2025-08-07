import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .pipelines import Pipeline, get_pipeline
from .datasets import AudioSample, get_dataset

__all__ = [
    'run_pipeline',
]


def run_pipeline(
    pipeline_name: str,
    dataset_names: list[str],
    root_dir: str | Path,
    max_samples: int | None = None,
):
    '''
    Runs a pipeline on a list of datasets. See asr_eval/bench/README.md for details.
    
    This function has also a command line interface:
    
    python -m asr_eval.bench.run -p PIPELINE [PIPELINE ...] \
        -d DATASET [DATASET ...] [-r ROOT_DIR] [-m MAX_SAMPLES]
    '''
    root_dir = Path(root_dir)
    
    # lazy pipeline instantiation
    pipeline_cls = get_pipeline(pipeline_name)
    pipeline_obj: Pipeline | None = None
    
    for dataset_name in dataset_names: 
        dir = root_dir / pipeline_name / dataset_name
        
        # try to skip the whole dataset if ready
        if (
            max_samples is not None
            and pipeline_cls.FILENAME
            and all(
                (dir / str(i) / pipeline_cls.FILENAME).exists()
                for i in range(max_samples)
            )
        ):
            continue
        
        dataset = get_dataset(dataset_name)()
        
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.take(max_samples)
        
        for i, sample in enumerate(cast(Sequence[AudioSample], dataset)):
            # try to skip sample if ready
            if (dir / str(i) / pipeline_cls.FILENAME).exists():
                continue
            
            # lazy pipeline instantiation
            pipeline_obj = pipeline_obj or pipeline_cls()
            
            print('running', pipeline_name, dataset_name, i)
            pipeline_obj.run_on_dataset_sample(
                dataset_name, i, sample, root_dir, dir / str(i)
            )


if __name__ == '__main__':
    # example: `python -m asr_eval.bench.run -p whisper-tiny -d podlodka -m 1`
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline', nargs='+', required=True, help='pipeline names')
    parser.add_argument('-d', '--dataset', nargs='+', required=True, help='dataset names')
    parser.add_argument('-r', '--root_dir', default='outputs', help='dir to save the results')
    parser.add_argument('-m', '--max_samples', type=int, required=False, help='max samples per dataset')
    args = parser.parse_args()
    
    for name in args.pipeline:
        run_pipeline(
            pipeline_name=name,
            dataset_names=args.dataset,
            root_dir=args.root_dir,
            max_samples=args.max_samples,
        )