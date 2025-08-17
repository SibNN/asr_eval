from __future__ import annotations

from collections.abc import Container, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
import pickle
from typing import Literal, cast

import pandas as pd
from tqdm.auto import tqdm

from .datasets import AudioSample, get_dataset, get_dataset_info
from ..align.alignment import Alignment, MultipleAlignment
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.transcription import SingleVariantTranscription, Transcription
from ..utils.serializing import load_from_json
from ..segments.segment import TimedText


__all__ = [
    'Evaluator',
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


class Evaluator:
    def __init__(self, cache_dir: str | Path = 'tmp/evaluator_cache'):
        self.cache_dir = Path(cache_dir)
        self.df = pd.DataFrame(
            columns=[f.name for f in fields(PredictionDataframeRow)]
        ).set_index('path') # type: ignore
    
    def get_multiple_alignments(
        self,
        dataset_name: str,
        pipeline_names: Container[str] | None = None,
    ) -> dict[int, MultipleAlignment]:
        is_unlabeled = get_dataset_info(dataset_name).unlabeled
        results: dict[int, MultipleAlignment] = {}
        if not is_unlabeled:
            for dataset_name, sample_idx, df_for_sample in self.groupby_sample(dataset_name=dataset_name):
                assert set(df_for_sample['aligned_against'].unique()) == {True} # type: ignore
                results[sample_idx] = MultipleAlignment(
                    baseline=self.get_ground_truth_cached(dataset_name, sample_idx),
                    alignments=dict(zip(
                        df_for_sample['pipeline_name'], # type: ignore
                        df_for_sample['alignment'], # type: ignore
                    )),
                )
        else:
            pass  # TODO support alignments for unlabeled datasets
        return results
    
    def groupby_sample(self, dataset_name: str | None = None) -> Iterator[tuple[str, int, pd.DataFrame]]:
        for _dataset_name, df_for_dataset in self.df.groupby('dataset_name'): # type: ignore
            _dataset_name = cast(str, _dataset_name)
            if dataset_name is None or _dataset_name == dataset_name:
                for sample_idx, df_for_sample in df_for_dataset.groupby('sample_idx'): # type: ignore
                    sample_idx = cast(int, sample_idx)
                    yield _dataset_name, sample_idx, df_for_sample

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
        # list predictions not loaded yet
        preds_on_disk = list_predictions(
            root_dir=root_dir,
            max_sample_idx=max_sample_idx,
            only_pipelines=only_pipelines,
            only_datasets=only_datasets,
            exclude_pipelines=exclude_pipelines,
            exclude_datasets=exclude_datasets,
        )
        preds_on_disk = [p for p in preds_on_disk if p.path not in self.df.index]
        
        # load predictions incrementally with progress bar
        preds_rows = [
            PredictionDataframeRow(
                **vars(pred_info),
                pred=LoadedPrediction.load_cached(pred_info.path),
            )
            for pred_info in tqdm(list(preds_on_disk))
        ]
        self.df = pd.concat([
            self.df,
            pd.DataFrame([vars(row) for row in preds_rows]).set_index('path'), # type: ignore
        ])
        
        # calculating alignments
        for dataset_name, sample_idx, df_for_sample in self.groupby_sample():
            is_unlabeled = get_dataset_info(dataset_name).unlabeled
            rows = [
                PredictionDataframeRow(path=cast(Path, path), **row.to_dict()) # type: ignore
                for path, row in df_for_sample.iterrows() # type: ignore
            ]
            if not is_unlabeled:
                for row in rows:
                    if row.alignment is None or row.aligned_against is not True:
                        true = self.get_ground_truth_cached(dataset_name, sample_idx)
                        cache_path = row.path.with_suffix('.align.pkl')
                        if cache_path.is_file():
                            alignment = pickle.loads(cache_path.read_bytes())
                        else:
                            print(f'Aligning: {dataset_name} #{sample_idx} truth VS {row.pipeline_name}')
                            alignment = Alignment.from_predictions(true=true, pred=row.pred.pred)
                            cache_path.write_bytes(pickle.dumps(alignment))
                        self.df.at[row.path, 'true'] = true # type: ignore
                        self.df.at[row.path, 'alignment'] = alignment # type: ignore
                        self.df.at[row.path, 'aligned_against'] = True # type: ignore
            else:
                # baseline_name = pref_baseline if pref_baseline in preds else sorted(preds)[0]
                pass  # TODO support alignments for unlabeled datasets
    
    def get_ground_truth_cached(self, dataset_name: str, sample_idx: int) -> Transcription:
        cache_path = self.cache_dir / dataset_name / str(sample_idx) / 'true.pkl'
        if cache_path.is_file():
            return pickle.loads(cache_path.read_bytes())
        else:
            result = load_ground_truth(dataset_name, sample_idx)
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            cache_path.write_bytes(pickle.dumps(result))
            return result

    
def load_ground_truth(dataset_name: str, sample_idx: int) -> Transcription:
    dataset = get_dataset(dataset_name)
    sample = cast(AudioSample, dataset[sample_idx])
    return parse_multivariant_string(sample['transcription'])
    

def list_predictions(
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