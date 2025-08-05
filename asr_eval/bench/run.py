from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .pipelines import Pipeline, pipelines_registry
from .datasets import AudioSample, datasets_registry


def run_pipeline(
    pipeline_name: str,
    dataset_names: list[str],
    root_dir: str | Path,
    max_samples: int | None = None,
):
    root_dir = Path(root_dir)
    
    # lazy pipeline instantiation
    pipeline_cls = pipelines_registry[pipeline_name]
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
        
        dataset = datasets_registry[dataset_name]().shuffle(0)
        
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.take(max_samples)
        
        for i, sample in enumerate(cast(Sequence[AudioSample], dataset)):
            # try to skip sample if ready
            if (dir / str(i) / pipeline_cls.FILENAME).exists():
                continue
            
            # lazy pipeline instantiation
            pipeline_obj = pipeline_obj or pipeline_cls()
            
            pipeline_obj.run_on_dataset_sample(
                dataset_name, i, sample, root_dir, dir / str(i)
            )