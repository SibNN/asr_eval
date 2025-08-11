from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from tqdm.auto import tqdm
from datasets import Dataset

from .datasets import AudioSample, get_dataset, get_dataset_info
from ..align.alignment import MultipleAlignment
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.transcription import SingleVariantTranscription, Transcription
from ..utils.serializing import load_from_json
from ..segments.segment import TimedText


__all__ = [
    'Evaluator',
]


# keys for evaluator storage:
SAMPLE_KEY = tuple[str, int]  # dataset_name, sample_idx
PIPELINE_KEY = str  # pipeline_name
    

@dataclass
class LoadedPrediction:
    pred: SingleVariantTranscription
    pred_timed: list[TimedText] | None


class Evaluator:
    '''
    An evaluator that loads the results of transcriber pipelines into a dataframe.
    
    TODO more detailed docs.
    '''
    
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.predictions: dict[SAMPLE_KEY, dict[PIPELINE_KEY, LoadedPrediction]] = defaultdict(dict)
        self.multple_alignments: dict[SAMPLE_KEY, MultipleAlignment] = {}
        
        # caches for ground truth
        self.ground_truths: dict[str, dict[int, Transcription]] = defaultdict(dict)
        self._datasets_cache: dict[str, Dataset] = {}
        
    def load_results(
        self,
        skip_loaded: bool = True,
        max_sample_idx: int | None = None,
    ) -> Self:
        # 1. update predictions
        paths: list[Path] = []
        for path in list(self.root_dir.glob('*/*/*/transcription.json')):
            pipeline_name, dataset_name, sample_idx, _ = path.relative_to(self.root_dir).parts
            sample_idx = int(sample_idx)
            
            if (
                skip_loaded
                and (dataset_name, sample_idx) in self.predictions
                and pipeline_name in self.predictions[dataset_name, sample_idx]
            ):
                continue
            if (
                max_sample_idx is not None
                and int(str(path.parent.name)) > max_sample_idx
            ):
                continue
            paths.append(path)
        
        for path in tqdm(paths):
            pipeline_name, dataset_name, sample_idx, _ = path.relative_to(self.root_dir).parts
            sample_idx = int(sample_idx)
            
            data = load_from_json(path)
            if data['type'] == 'timed_transcription':
                timed_transcription = data['output']
                transcription = ' '.join(seg.text for seg in timed_transcription)
            else:
                timed_transcription = None
                transcription = data['output']
            
            self.predictions[dataset_name, sample_idx][pipeline_name] = LoadedPrediction(
                pred=parse_single_variant_string(transcription),
                pred_timed=timed_transcription,
            )
            
        # 2. update multiple alignments
        for (dataset_name, sample_idx), predictions in tqdm(self.predictions.items()):
            if len(predictions) == 0:
                continue
            multiple_alignment = self.multple_alignments.get((dataset_name, sample_idx), None)
            if get_dataset_info(dataset_name).unlabeled:
                # for unlabeled dataset, use one of the predictions as a baseline
                if multiple_alignment is None:
                    baseline_name = sorted(predictions)[0]
                    self.multple_alignments[dataset_name, sample_idx] = multiple_alignment = (
                        MultipleAlignment(baseline=predictions[baseline_name].pred)
                    )
                else:
                    baseline_name = multiple_alignment.baseline_name
                    assert isinstance(baseline_name, str)
                for pipeline_name, loaded_prediction in predictions.items():
                    if baseline_name != baseline_name:
                        multiple_alignment.add_alignment_from_prediction(
                            pipeline_name, loaded_prediction.pred
                        )
            else:
                # for labeled dataset, use ground truth as a baseline
                if multiple_alignment is None:
                    self.multple_alignments[dataset_name, sample_idx] = multiple_alignment = (
                        MultipleAlignment(baseline=self.get_ground_truth(dataset_name, sample_idx))
                    )
                for pipeline_name, loaded_prediction in predictions.items():
                    multiple_alignment.add_alignment_from_prediction(
                        pipeline_name, loaded_prediction.pred
                    )
        
        return self
    
    def _get_dataset(self, dataset_name: str) -> Dataset:
        if dataset_name not in self._datasets_cache:
            print(f'Loading dataset {dataset_name}')
            self._datasets_cache[dataset_name] = get_dataset(dataset_name)()
            print(f'Loaded dataset {dataset_name}')
        return self._datasets_cache[dataset_name]
    
    def get_ground_truth(
        self, dataset_name: str, sample_idx: int
    ) -> Transcription:
        if sample_idx not in self.ground_truths[dataset_name]:
            dataset = self._get_dataset(dataset_name)
            sample = cast(AudioSample, dataset[sample_idx])
            self.ground_truths[dataset_name][sample_idx] = (
                parse_multivariant_string(sample['transcription'])
            )
        return self.ground_truths[dataset_name][sample_idx]
    