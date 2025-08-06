from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Self, TypedDict, cast
import pickle

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset

from .datasets import AudioSample, get_dataset
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.recursive import align
from ..align.data import MatchesList, Token, MultiVariant
from ..utils.serializing import load_from_json
from ..segments.segment import TimedText


GROUND_TRUTH_TYPE = tuple[str, list[Token | MultiVariant]]


class Evaluator:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.df = self._predictions_to_df({})
        
        # caches
        self.ground_truths: dict[str, dict[int, GROUND_TRUTH_TYPE]] = defaultdict(dict)
        self._datasets_cache: dict[str, Dataset] = {}
        
    def load_results(self, skip_loaded: bool = True) -> Self:
        new_df = self._predictions_to_df({
            path: self._load_result(path)
            for path in tqdm(list(self.root_dir.glob('*/*/*/transcription.json')))
            if not skip_loaded or path not in self.df.index
        })
        self.df = pd.concat([self.df, new_df])
        return self
    
    def get_ground_truth(
        self, dataset_name: str, sample_idx: int
    ) -> GROUND_TRUTH_TYPE:
        if sample_idx not in self.ground_truths[dataset_name]:
            dataset = self._get_dataset(dataset_name)
            sample = cast(AudioSample, dataset[sample_idx])
            self.ground_truths[dataset_name][sample_idx] = (
                sample['transcription'], parse_multivariant_string(sample['transcription'])
            )
        return self.ground_truths[dataset_name][sample_idx]
    
    def _predictions_to_df(self, predictions: dict[Path, _SamplePrediction]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path, pred in predictions.items():
            rows.append(row := cast(dict[str, Any], pred.copy()))
            row['path'] = path
            row['ground_truth'], row['ground_truth_words'] = (
                self.get_ground_truth(pred['dataset_name'], pred['sample_idx'])
            )
        df = pd.DataFrame(columns=(
            ['path']
            + list(_SamplePrediction.__required_keys__)
            + ['ground_truth', 'ground_truth_words']
        ), data=rows)
        df = df.set_index('path') # type: ignore
        return df

    def _load_result(self, path: Path) -> _SamplePrediction:
        _, dataset_name, sample_idx, _ = path.relative_to(self.root_dir).parts
        _, ground_truth_words = self.get_ground_truth(dataset_name, int(sample_idx))
        return sample_prediction_from_file(path, ground_truth_words)
    
    def _get_dataset(self, dataset_name: str) -> Dataset:
        if dataset_name not in self._datasets_cache:
            self._datasets_cache[dataset_name] = get_dataset(dataset_name)()
        return self._datasets_cache[dataset_name]


class _SamplePrediction(TypedDict):
    '''A result of running a Transcriber on a single sample'''
    pipeline_name: str
    dataset_name: str
    sample_idx: int
    transcription: str
    transcription_words: list[Token]
    alignment: MatchesList
    timed_transcription: list[TimedText] | None
    

def sample_prediction_from_file(
    path: Path, ground_truth_words: list[Token | MultiVariant]
) -> _SamplePrediction:
    if (pkl_path := path.with_suffix('.pkl')).exists():
        return pickle.loads(pkl_path.read_bytes())
    else:
        data = load_from_json(path)
        if data['type'] == 'timed_transcription':
            timed_transcription = data['output']
            transcription = ' '.join(seg.text for seg in timed_transcription)
        else:
            timed_transcription = None
            transcription = data['output']
        
        transcription_words = parse_single_variant_string(transcription)
        alignment = align(ground_truth_words, transcription_words)
        
        obj: _SamplePrediction = {
            'pipeline_name': path.parts[-4],
            'dataset_name': path.parts[-3],
            'sample_idx': int(path.parts[-2]),
            'transcription': transcription,
            'transcription_words': transcription_words,
            'alignment': alignment,
            'timed_transcription': timed_transcription,
        }
        
        pkl_path.write_bytes(pickle.dumps(obj))
        return obj