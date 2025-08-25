from __future__ import annotations

from collections.abc import Container, Iterator
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Literal, cast

from tqdm.auto import tqdm

from ..utils.dataframe import DataclassDataFrame
from .datasets import AudioSample, get_dataset, get_dataset_info
from ..align.alignment import Alignment, MultipleAlignment
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.transcription import SingleVariantTranscription, Transcription
from ..utils.serializing import load_from_json
from ..segments.segment import TimedText


__all__ = [
    'PredictionLoader',
    'PredictionInfo',
    'LoadedPrediction',
    'PredictionDataframeRow',
]
    

@dataclass
class PredictionInfo:
    dataset_name: str
    sample_idx: int
    pipeline_name: str
    path: Path


@dataclass
class LoadedPrediction:
    pred: SingleVariantTranscription
    pred_timed: list[TimedText] | None
    elapsed_time: float
    
    @classmethod
    def load(cls, filepath: Path) -> LoadedPrediction:
        data = load_from_json(filepath)
        if data['type'] == 'timed_transcription':
            timed_transcription = data['output']
            transcription = ' '.join(seg.text for seg in timed_transcription)
        else:
            timed_transcription = None
            transcription = data['output']
        return cls(
            pred=parse_single_variant_string(transcription),
            pred_timed=timed_transcription,
            elapsed_time=data.get('elapsed_time', float('nan')),
        )

    @classmethod
    def load_cached(cls, filepath: Path) -> LoadedPrediction:
        cache_path = filepath.with_suffix('.pkl')
        if cache_path.is_file():
            return pickle.loads(cache_path.read_bytes())
        else:
            pred = cls.load(filepath)
            cache_path.write_bytes(pickle.dumps(pred))
            return pred


@dataclass
class PredictionDataframeRow:
    dataset_name: str
    sample_idx: int
    pipeline_name: str
    path: Path
    pred: LoadedPrediction
    true: Transcription | None = None
    alignment: Alignment | None = None
    aligned_against: str | Literal[True] = True


class PredictionLoader:
    def __init__(self, cache_dir: str | Path = 'tmp/evaluator_cache'):
        self.cache_dir = Path(cache_dir)
        self.df = DataclassDataFrame[PredictionDataframeRow]()
    
    def list_datasets(self) -> list[str]:
        return list(set(self.df['dataset_name']))
    
    def list_pipelines(self) -> list[str]:
        return list(set(self.df['pipeline_name']))

    def get_prediction(
        self,
        dataset_name: str,
        sample_idx: int,
        pipeline_name: str,
    ) -> LoadedPrediction:
        for row in self.df.data:
            if (
                row.dataset_name == dataset_name
                and row.sample_idx == sample_idx
                and row.pipeline_name == pipeline_name
            ):
                return row.pred
        raise AssertionError(f'Cannot find a row for {dataset_name}, {sample_idx}, {pipeline_name}')
            
    def get_multiple_alignments(
        self,
        dataset_name: str,
        pipeline_names: Container[str] | None = None,
    ) -> dict[int, MultipleAlignment]:
        results: dict[int, MultipleAlignment] = {}
        if not get_dataset_info(dataset_name).unlabeled:
            for sample_idx, df_for_sample in (
                self.df
                .sort_values('sample_idx')
                .filter(dataset_name=dataset_name)
                .groupby('sample_idx')
            ):
                results[sample_idx] = MultipleAlignment(
                    baseline=self.get_ground_truth_cached(dataset_name, sample_idx),
                    alignments={
                        row.pipeline_name: cast(Alignment, row.alignment)
                        for row in df_for_sample.data
                        if pipeline_names is None or row.pipeline_name in pipeline_names
                    },
                )
        else:
            pass  # TODO support alignments for unlabeled datasets
        return results

    def load_results(
        self,
        root_dir: str | Path = 'outputs',
        max_sample_idx: int | None = None,
        only_pipelines: Container[str] | None = None,
        only_datasets: Container[str] | None = None,
        pref_baseline: str | None = None,  # TODO for unlabeled data
        exclude_pipelines: Container[str] = (),
        exclude_datasets: Container[str] = (),
        with_relabelings: bool = True,
    ):
        # list predictions not loaded yet
        preds_on_disk = _list_predictions(
            root_dir=root_dir,
            max_sample_idx=max_sample_idx,
            only_pipelines=only_pipelines,
            only_datasets=only_datasets,
            exclude_pipelines=exclude_pipelines,
            exclude_datasets=exclude_datasets,
        )
        preds_on_disk = [
            p for p in preds_on_disk
            if p.path not in set(self.df['path'])
        ]
        
        # load predictions incrementally with progress bar
        for pred_info in tqdm(list(preds_on_disk)):
            self.df.data.append(PredictionDataframeRow(
                **vars(pred_info),
                pred=LoadedPrediction.load_cached(pred_info.path),
            ))
        
        # calculating alignments
        for (dataset_name, sample_idx), df_for_sample in self.df.groupby(['dataset_name', 'sample_idx']):
            if not get_dataset_info(dataset_name).unlabeled:
                for row in df_for_sample.data:
                    if row.alignment is None:
                        row.true = self.get_ground_truth_cached(dataset_name, sample_idx)
                        cache_path = row.path.with_suffix('.align.pkl')
                        if cache_path.is_file():
                            row.alignment = pickle.loads(cache_path.read_bytes())
                        else:
                            print(f'Aligning: {dataset_name} #{sample_idx} truth VS {row.pipeline_name}')
                            row.alignment = Alignment.from_predictions(true=row.true, pred=row.pred.pred)
                            cache_path.write_bytes(pickle.dumps(row.alignment))
            else:
                # baseline_name = pref_baseline if pref_baseline in preds else sorted(preds)[0]
                pass  # TODO support alignments for unlabeled datasets
    
    def get_ground_truth_cached(self, dataset_name: str, sample_idx: int) -> Transcription:
        cache_path = self.cache_dir / dataset_name / str(sample_idx) / 'true.pkl'
        if cache_path.is_file():
            return pickle.loads(cache_path.read_bytes())
        else:
            result = _load_ground_truth(dataset_name, sample_idx)
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            cache_path.write_bytes(pickle.dumps(result))
            return result

    
def _load_ground_truth(dataset_name: str, sample_idx: int) -> Transcription:
    dataset = get_dataset(dataset_name)
    sample = cast(AudioSample, dataset[sample_idx])
    return parse_multivariant_string(sample['transcription'])
    

def _list_predictions(
    root_dir: str | Path,
    max_sample_idx: int | None = None,
    only_pipelines: Container[str] | None = None,
    only_datasets: Container[str] | None = None,
    exclude_pipelines: Container[str] = (),
    exclude_datasets: Container[str] = (),
) -> Iterator[PredictionInfo]:
    root_dir = Path(root_dir)
    for path in sorted(root_dir.glob('*/*/*/transcription.json')):
        pipeline_name, dataset_name, sample_idx, _ = path.relative_to(root_dir).parts
        sample_idx = int(sample_idx)
        if exclude_pipelines in exclude_pipelines:
            continue
        if dataset_name in exclude_datasets:
            continue
        if max_sample_idx is not None and sample_idx > max_sample_idx:
            continue
        if only_datasets is not None and dataset_name not in only_datasets:
            continue
        if only_pipelines is not None and pipeline_name not in only_pipelines:
            continue
        yield PredictionInfo(
            dataset_name=dataset_name,
            sample_idx=sample_idx,
            pipeline_name=pipeline_name,
            path=path,
        )