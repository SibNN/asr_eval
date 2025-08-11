from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Self, TypedDict, cast
import pickle

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset

from .datasets import AudioSample, get_dataset
from ..align.alignment import Alignment
from ..align.parsing import parse_single_variant_string, parse_multivariant_string
from ..align.transcription import MultiVariantTranscription, SingleVariantTranscription
from ..utils.serializing import load_from_json
from ..segments.segment import TimedText


__all__ = [
    'Evaluator',
]


class Evaluator:
    '''
    An evaluator that loads the results of transcriber pipelines into a dataframe.
    
    TODO more detailed docs.
    '''
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.df = pd.DataFrame(
            columns=list(_EvaluatorDataframeRow.__required_keys__)
        ).set_index('path') # type: ignore
        
        # caches
        self.ground_truths: dict[
            str, dict[int, MultiVariantTranscription | SingleVariantTranscription]
        ] = defaultdict(dict)
        self._datasets_cache: dict[str, Dataset] = {}
        
    def load_results(self, skip_loaded: bool = True, max_sample_idx: int | None = None) -> Self:
        files = [
            path for path in list(self.root_dir.glob('*/*/*/transcription.json'))
            # skip if already loaded, if `skip_loaded=True`
            if (not skip_loaded or path not in self.df.index)
            # skip if sample index is larger than `max_sample_idx`
            and (max_sample_idx is None or int(str(path.parent.name)) <= max_sample_idx)
        ]
        
        df_rows: list[_EvaluatorDataframeRow] = []
        for path in tqdm(files):
            pipeline_name, dataset_name, sample_idx, _ = path.relative_to(self.root_dir).parts
            sample_idx = int(sample_idx)
            result = _TranscriberPipelineResult.from_file(
                path,
                get_true=partial(self.get_ground_truth, dataset_name, sample_idx),
            )
            df_rows.append({
                'path': path,
                'pipeline_name': pipeline_name,
                'dataset_name': dataset_name,
                'sample_idx': sample_idx,
                'ground_truth': result.true,
                'pred': result.pred,
                'pred_timed': result.pred_timed,
                'alignment': result.alignment,
            })
        
        if len(df_rows):
            self.df = pd.concat([
                self.df,
                pd.DataFrame(data=df_rows).set_index('path'), # type: ignore
            ]).groupby(level=0).last() # type: ignore
        
        return self
    
    def _get_dataset(self, dataset_name: str) -> Dataset:
        if dataset_name not in self._datasets_cache:
            print(f'Loading dataset {dataset_name}')
            self._datasets_cache[dataset_name] = get_dataset(dataset_name)()
            print(f'Loaded dataset {dataset_name}')
        return self._datasets_cache[dataset_name]
    
    def get_ground_truth(
        self, dataset_name: str, sample_idx: int
    ) -> MultiVariantTranscription | SingleVariantTranscription:
        if sample_idx not in self.ground_truths[dataset_name]:
            dataset = self._get_dataset(dataset_name)
            sample = cast(AudioSample, dataset[sample_idx])
            self.ground_truths[dataset_name][sample_idx] = (
                parse_multivariant_string(sample['transcription'])
            )
        return self.ground_truths[dataset_name][sample_idx]
    

class _EvaluatorDataframeRow(TypedDict):
    path: Path
    pipeline_name: str
    dataset_name: str
    sample_idx: int
    ground_truth: MultiVariantTranscription | SingleVariantTranscription
    pred: SingleVariantTranscription
    pred_timed: list[TimedText] | None
    alignment: Alignment


@dataclass
class _TranscriberPipelineResult:
    '''
    A result of running a TranscriberPipeline on a single sample. Keeps the
    transcription of a single dataset sample and its alignment with ground truth.
    
    .from_file() will load a json file with fields "transcription" or "timed_transcription",
    perform alignment with the given ground truth and return the results.
    
    Will cache the results of {file}.json to a neighbour file {file}.pkl, or
    will load .pkl if it exists.
    
    ground truth is callable for lazy evaluation, since it is required only if .pkl was not found.
    '''
    true: MultiVariantTranscription | SingleVariantTranscription
    pred: SingleVariantTranscription
    pred_timed: list[TimedText] | None
    alignment: Alignment
    
    @classmethod
    def from_file(
        cls,
        path: Path,
        get_true: Callable[[], MultiVariantTranscription | SingleVariantTranscription],
    ) -> Self:
        assert path.suffix == '.json'
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
                
            true = get_true()
            
            pred = parse_single_variant_string(transcription)
            alignment = Alignment.from_predictions(true=true, pred=pred)
            
            result = cls(
                true=true,
                pred=pred,
                pred_timed=timed_transcription,
                alignment=alignment,
            )
            
            pkl_path.write_bytes(pickle.dumps(result))
            return result