from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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


class Evaluator:
    def __init__(self, cache_dir: str | Path = 'tmp/evaluator_cache'):
        cache_dir = Path(cache_dir)
        self._truths = TupleKeyShelf(cache_dir / 'truths.db')
        self._predictions = TupleKeyShelf(cache_dir / 'predictions.db')
        self._alignments = TupleKeyShelf(cache_dir / 'alignments.db')
        self._baseline_names: dict[tuple[str, str], str] = {}
    
    def list_datasets(self) -> list[str]:
        return sorted(set([dataset_name for dataset_name, _, _ in self._predictions]))
    
    def list_pipelines(self) -> list[str]:
        return sorted(set([pipeline_name for _, _, pipeline_name in self._predictions]))
    
    def get_prediction(self, dataset_name: str, sample_idx: str, pipeline_name: str) -> LoadedPrediction:
        return self._predictions[dataset_name, sample_idx, pipeline_name]
    
    def get_ground_truth(self, dataset_name: str, sample_idx: str) -> Transcription:
        key = (dataset_name, sample_idx)
        try:
            return self._truths[key]
        except KeyError:
            dataset = get_dataset(dataset_name)
            sample = cast(AudioSample, dataset[int(sample_idx)])
            self._truths[key] = result = parse_multivariant_string(sample['transcription'])
            return result

    def group_predictions_by_sample(self) -> dict[tuple[str, str], dict[str, LoadedPrediction]]:
        result: dict[tuple[str, str], dict[str, LoadedPrediction]] = defaultdict(dict)
        for (dataset_name, sample_idx, pipeline_name), pred in self._predictions.items():
            result[dataset_name, sample_idx][pipeline_name] = pred
        return result
    
    def get_multiple_alignments(
        self,
        dataset_name: str,
        pipeline_names: Container[str] | None = None,
    ) -> dict[str, MultipleAlignment]:
        grouped_alignments: dict[str, dict[tuple[str, str], Alignment]] = defaultdict(dict)
        for (
            (_dataset_name, sample_idx, pipeline_name, baseline_name), alignment
        ) in self._alignments.items():
            if _dataset_name == dataset_name and (pipeline_names is None or pipeline_name in pipeline_names):
                grouped_alignments[sample_idx][pipeline_name, baseline_name] = alignment
        grouped_alignments = dict(sorted(grouped_alignments.items(), key=lambda item: int(item[0])))
        
        results: dict[str, MultipleAlignment] = {}
        if get_dataset_info(dataset_name).unlabeled:
            for sample_idx, available_alignments in grouped_alignments.items():
                baseline_name = self._baseline_names[dataset_name, sample_idx]
                results[sample_idx] = MultipleAlignment(
                    baseline=self._predictions[dataset_name, sample_idx, baseline_name],
                    baseline_name=baseline_name,
                    alignments={
                        pipeline_name: alignment
                        for (pipeline_name, _baseline_name), alignment in available_alignments.items()
                        if _baseline_name == baseline_name
                    }
                )
        else:
            for sample_idx, available_alignments in grouped_alignments.items():
                results[sample_idx] = MultipleAlignment(
                    baseline=self.get_ground_truth(dataset_name, sample_idx),
                    baseline_name=True,
                    alignments={
                        pipeline_name: alignment
                        for (pipeline_name, _baseline_name), alignment in available_alignments.items()
                        if _baseline_name == ''  # aligned against ground truth
                    }
                )
        return results

    def load_results(
        self,
        root_dir: str | Path = 'outputs',
        max_sample_idx: int | None = None,
        only_pipelines: Container[str] | None = None,
        only_datasets: Container[str] | None = None,
        pref_baseline: str | None = None,
        exclude_pipelines: Container[str] = (),
        exclude_datasets: Container[str] = (),
    ):
        # load predictions from json files to shelf
        preds_on_disk = list_predictions(
            root_dir,
            max_sample_idx=max_sample_idx,
            only_pipelines=only_pipelines,
            only_datasets=only_datasets,
            exclude_pipelines=exclude_pipelines,
            exclude_datasets=exclude_datasets,
        )
        for dataset_name, sample_idx, pipeline_name, json_path in tqdm(list(preds_on_disk)):
            key = (dataset_name, sample_idx, pipeline_name)
            if key not in self._predictions:
                self._predictions[key] = load_prediction(json_path)
        
        # calculate alignments
        for (dataset_name, sample_idx), preds in self.group_predictions_by_sample().items():
            if get_dataset_info(dataset_name).unlabeled:
                # unlabeled dataset
                baseline_name = pref_baseline if pref_baseline in preds else sorted(preds)[0]
                self._baseline_names[dataset_name, sample_idx] = baseline_name
                for pipeline_name, pred in preds.items():
                    if pipeline_name != baseline_name:
                        key = (dataset_name, sample_idx, pipeline_name, baseline_name)
                        if not key in self._alignments:
                            print(
                                f'Aligning: {dataset_name} #{sample_idx} {baseline_name} VS {pipeline_name}'
                            )
                            self._alignments[key] = Alignment.from_predictions(
                                preds[baseline_name].pred, pred.pred
                            )
            else:
                # labeled dataset
                for pipeline_name, pred in preds.items():
                    key = (dataset_name, sample_idx, pipeline_name, '')
                    if not key in self._alignments:
                        print(
                            f'Aligning: {dataset_name} #{sample_idx} truth VS {pipeline_name}'
                        )
                        self._alignments[key] = Alignment.from_predictions(
                            self.get_ground_truth(dataset_name, sample_idx), pred.pred
                        )


@dataclass
class LoadedPrediction:
    pred: SingleVariantTranscription
    pred_timed: list[TimedText] | None
    elapsed_time: float


def load_prediction(filepath: Path) -> LoadedPrediction:
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
    

def list_predictions(
    root_dir: str | Path,
    max_sample_idx: int | None = None,
    only_pipelines: Container[str] | None = None,
    only_datasets: Container[str] | None = None,
    exclude_pipelines: Container[str] = (),
    exclude_datasets: Container[str] = (),
) -> Iterator[tuple[str, str, str, Path]]:
    root_dir = Path(root_dir)
    for path in sorted(root_dir.glob('*/*/*/transcription.json')):
        pipeline_name, dataset_name, sample_idx, _ = path.relative_to(root_dir).parts
        if exclude_pipelines in exclude_pipelines:
            continue
        if dataset_name in exclude_datasets:
            continue
        if max_sample_idx is not None and int(sample_idx) > max_sample_idx:
            continue
        if only_datasets is not None and dataset_name not in only_datasets:
            continue
        if only_pipelines is not None and pipeline_name not in only_pipelines:
            continue
        yield dataset_name, sample_idx, pipeline_name, path