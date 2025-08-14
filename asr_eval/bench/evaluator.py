from __future__ import annotations

from collections import defaultdict
from collections.abc import Container
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from tqdm.auto import tqdm

from .datasets import AudioSample, get_dataset, get_dataset_info
from ..align.alignment import Alignment, MultipleAlignment
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.transcription import SingleVariantTranscription, Transcription
from ..utils.serializing import load_from_json
from ..utils.shelves import TupleKeyShelf
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
    elapsed_time: float


class Evaluator:
    '''
    An evaluator that loads the results of transcriber pipelines into a dataframe.
    
    TODO more detailed docs.
    '''
    
    def __init__(self, root_dir: str | Path, cache_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.cache_dir = Path(cache_dir)
        
        self.predictions: dict[SAMPLE_KEY, dict[PIPELINE_KEY, LoadedPrediction]] = defaultdict(dict)
        self.multiple_alignments: dict[SAMPLE_KEY, MultipleAlignment] = {}
        
        # disk cache for tokenized ground truth, tokenized predictions and alignments
        # key format for ground truth:
        # - (dataset_name, str(sample_idx))
        # key format for prediction:
        # - (pipeline_name, dataset_name, str(sample_idx))
        # key format for alignment against ground truth:
        # - (pipeline_name, dataset_name, str(sample_idx), 'true')
        # key format for alignment against baseline:
        # - (pipeline_name, dataset_name, str(sample_idx), 'baseline', baseline_pipeline_name)
        self._shelf = TupleKeyShelf(self.cache_dir / 'evaluator_cache')
        
        # TODO
        # first make all alignments, then group into multiple alignments
        # paralellize making alignments with multithreading
        # rewrite alignment in C
    
    def list_datasets(self) -> list[str]:
        return sorted([dataset_name for dataset_name, _sample_idx in self.predictions])
    
    def list_pipelines(self) -> list[str]:
        pipelines: set[str] = set()
        for preds in self.predictions.values():
            pipelines |= set(preds.keys())
        return sorted(pipelines)
        
    def load_results(
        self,
        skip_loaded: bool = True,
        max_sample_idx: int | None = None,
        dataset_names: Container[str] | None = None,
        pref_baseline: str | None = None,
    ) -> Self:
        # 1. list available predictions
        paths: list[Path] = []
        for path in sorted(self.root_dir.glob('*/*/*/transcription.json')):
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
                and sample_idx > max_sample_idx
            ):
                continue
            if (
                dataset_names is not None
                and dataset_name not in dataset_names
            ):
                continue
            paths.append(path)
        
        # 2. load predictions
        for path in tqdm(paths):
            pipeline_name, dataset_name, sample_idx, _ = path.relative_to(self.root_dir).parts
            sample_idx = int(sample_idx)
            self.predictions[dataset_name, sample_idx][pipeline_name] = (
                self._get_prediction(pipeline_name, dataset_name, sample_idx, path)
            )
            
        # 2. load or update multiple alignments
        for (dataset_name, sample_idx), predictions in tqdm(self.predictions.items()):
            if len(predictions) == 0:
                continue
            multiple_alignment = self.multiple_alignments.get((dataset_name, sample_idx), None)
            
            if get_dataset_info(dataset_name).unlabeled:
                # for unlabeled dataset, use one of the predictions as a baseline
                if multiple_alignment is None:
                    baseline_name = (
                        sorted(predictions)[0]
                        if pref_baseline is None or pref_baseline not in predictions
                        else pref_baseline
                    )
                    self.multiple_alignments[dataset_name, sample_idx] = multiple_alignment = (
                        MultipleAlignment(
                            baseline=predictions[baseline_name].pred,
                            baseline_name=baseline_name,
                        )
                    )
                else:
                    baseline_name = multiple_alignment.baseline_name
                    assert isinstance(baseline_name, str)
                predictions = predictions.copy()
                predictions.pop(baseline_name)
                align_against_key = ('baseline', baseline_name,)
            
            else:
                # for labeled dataset, use ground truth as a baseline
                if multiple_alignment is None:
                    self.multiple_alignments[dataset_name, sample_idx] = multiple_alignment = (
                        MultipleAlignment(baseline=self.get_ground_truth(dataset_name, sample_idx))
                    )
                align_against_key = ('true',)
            
            # calculating alignments of predictions against the baseline
            for pipeline_name, loaded_prediction in predictions.items():
                key = (pipeline_name, dataset_name, str(sample_idx), *align_against_key)
                try:
                    multiple_alignment.alignments[pipeline_name] = self._shelf[key]
                except KeyError:
                    print(
                        f'Aligning: {dataset_name} #{sample_idx}'
                        f' {" ".join(align_against_key)} VS {pipeline_name}'
                    )
                    multiple_alignment.alignments[pipeline_name] = self._shelf[key] = (
                        Alignment.from_predictions(multiple_alignment.baseline, loaded_prediction.pred)
                    )
        
        return self
    
    def _get_prediction(
        self, pipeline_name: str, dataset_name: str, sample_idx: int, filepath: Path,
    ) -> LoadedPrediction:
        key = (pipeline_name, dataset_name, str(sample_idx))
        if (result := self._shelf.get(key, None)) is not None:
            return cast(LoadedPrediction, result)
        else:
            result = self._shelf[key] = self._load_prediction_from_json(filepath)
            return result
    
    def get_ground_truth(
        self, dataset_name: str, sample_idx: int
    ) -> Transcription:
        key = (dataset_name, str(sample_idx))
        if (result := self._shelf.get(key, None)) is not None:
            return cast(Transcription, result)
        else:
            result = self._shelf[key] = self._load_truth_from_dataset(dataset_name, sample_idx)
            return result
    
    def _load_prediction_from_json(self, filepath: Path) -> LoadedPrediction:
            data = load_from_json(filepath)
            if data['type'] == 'timed_transcription':
                timed_transcription = data['output']
                transcription = ' '.join(seg.text for seg in timed_transcription)
            else:
                timed_transcription = None
                transcription = data['output']
            return LoadedPrediction(
                pred=parse_single_variant_string(transcription),
                pred_timed=timed_transcription,
                elapsed_time=data.get('elapsed_time', float('nan')),
            )
    
    def _load_truth_from_dataset(self, dataset_name: str, sample_idx: int) -> Transcription:
        dataset = get_dataset(dataset_name)
        sample = cast(AudioSample, dataset[sample_idx])
        return parse_multivariant_string(sample['transcription'])