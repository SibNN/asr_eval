
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
import multiprocessing

import numpy as np
from wcmatch.fnmatch import fnmatch, NEGATE, SPLIT

if TYPE_CHECKING:
    import polars as pl

from asr_eval.bench.parsers import get_parser
from asr_eval.align.transcription import Transcription
from asr_eval.bench.datasets import get_dataset
from asr_eval.bench.datasets._registry import _get_transcription_by_sample_id # pyright: ignore[reportPrivateUsage]
from asr_eval.bench.pipelines import get_pipeline_index
from asr_eval.utils.storage.base_storage import BaseStorage
from asr_eval.align.alignment import Alignment, MultipleAlignment
from asr_eval.bench.datasets.dataset_spec import DatasetSpec


__all__ = [
    'PredictionLoader',
    'GroupKey',
    'SamplePrediction',
]


@dataclass(frozen=True)
class GroupKey:
    """A key to group predictions in
    :class:`~asr_eval.bench.loader.PredictionLoader`.
    """
    pipeline_name: str
    dataset_name: str
    augmentor: str
    parser: str

@dataclass(frozen=True)
class SamplePrediction:
    """A value to group predictions in
    :class:`~asr_eval.bench.loader.PredictionLoader`.
    """
    text: str
    elapsed_time: float

class PredictionLoader:
    """Loads and aligns predictions saved with
    :func:`~asr_eval.bench.run.run_pipeline`.
    
    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.
    
    Args:
        storage: A storage where the predictions were saved, typically a
            :class:`~asr_eval.utils.storage.shelf_storage.ShelfStorage`.
        cache: A cache to store alignments and other data to cache, may
            be initially filled or empty. The cache is reusable.
        pipelines: A list of pipelines names or patterns to load. By
            default loads all pipelines.
        dataset_specs: A list of dataset names, patterns or specs to
            load. By default loads all datasets. In simple case just
            use dataset name, such as :code:`dataset_specs=['fleurs']`.
            For more complex case, see the example below.
            
    Dataset specs (specificators with semicolons) allow to specify
    augmentors, parsers or sample count to load, see
    :doc:`/guide_evaluation_dashboard` for details. Examples:
    
    Example:
        :code:`PredictionLoader(dataset_specs=["fleurs:n=100!"])` will
        search for the fleurs dataset in the storage. For every "key"
        consisting of (pipeline + augmentor + parser) it will try to
        load exactly 100 first samples of the fleurs dataset. Will drop
        keys that have not all of these samples. Will drop all other
        samples. This ensures that for all the "keys" exactly the same
        sample set is loaded, which allows to compare them on the same
        data.
    """
    
    grouped_loaded_predictions: dict[GroupKey, dict[int, SamplePrediction]]
    """A public attribute that exposes a mapping. The keys are
    combinations of dataset + pipeline + augmentor + parser. The values
    are mapping from sample id to a prediction that keeps the predicted
    text and the inference time.
    """
    
    def __init__(
        self,
        storage: BaseStorage,
        cache: BaseStorage,
        pipelines: Sequence[str] = ('*',),
        dataset_specs: Sequence[str | DatasetSpec] = ('*',),
    ):
        self.storage = storage
        self.cache = cache
        
        dataset_specs = [
            DatasetSpec.from_string(x) if isinstance(x, str) else x
            for x in dataset_specs
        ]
        
        print('Listing all the available data')
        preds = storage.list_all(load_values=True, artifact_type='text')
        assert 'pipeline_name' in preds and 'dataset_name' in preds, (
            'No data found'
        )
        
        # sort values
        cols = preds.columns
        cols.remove('value')
        preds = preds.sort(cols)
        
        # TODO support hyperparameters

        self.grouped_loaded_predictions = {}
        
        for (pipeline_name, dataset_name, augmentor_name), preds_chunk in (
            preds.group_by(('pipeline_name', 'dataset_name', 'augmentor'))
        ):
            # check if matches the requested DatasetSpec-s
            dataset_spec = self._does_match_any_of_dataset_specs(
                dataset_name, augmentor_name, dataset_specs
            )
            if dataset_spec is None:
                continue
            
            # check if matches the requested pipelines
            if not fnmatch(pipeline_name, pipelines, flags=NEGATE | SPLIT):
                continue
            
            # filter samples or skip, according to dataset spec
            preds_chunk = self._get_requested_n_samples(
                preds_chunk, pipeline_name, dataset_name, dataset_spec
            )
            if preds_chunk is None:
                continue
            
            
            # storing the chunk
            for parser in dataset_spec.parser.split(','):
                group_key = GroupKey(
                    pipeline_name=pipeline_name,
                    dataset_name=dataset_name,
                    augmentor=augmentor_name,
                    parser=parser,
                )
                preds_group: dict[int, SamplePrediction] = {}
                for row in preds_chunk.iter_rows(named=True):
                    pred_text = row.pop('value')
                    time_key = row | {'artifact_type': 'elapsed_time'}
                    preds_group[row['sample_id']] = SamplePrediction(
                        text=pred_text,
                        elapsed_time=(
                            storage.get_row(**time_key)
                            if storage.has_row(**time_key)
                            else np.nan
                        )
                    )
                self.grouped_loaded_predictions[group_key] = preds_group

                self._calculate_and_store_alignments(group_key)
    
    def _calculate_and_store_alignments(self, group_key: GroupKey):
        print(f'Checking alignments for {asdict(group_key)}...', flush=True)
        
        preds = self.grouped_loaded_predictions[group_key]
        
        _do_align_args: list[tuple[
            dict[str, Any],  # row in alignment cache to store
            dict[str, Any],  # the requested operands for align operation
        ]] = []
        
        for sample_id, pred_info in preds.items():
            keys = {
                **asdict(group_key),
                'sample_id': sample_id,
                'artifact_type': 'alignment'
            }
            if not self.cache.has_row(**keys):
                print(
                    f'Parsing {asdict(group_key)} sample_id={sample_id}',
                    flush=True
                )
                pred = (
                    get_parser(group_key.parser, 'pred')
                    .parse_single_variant_transcription(pred_info.text)
                )
                true = self.get_annotation(
                    dataset_name=group_key.dataset_name,
                    parser_name=group_key.parser,
                    sample_id=sample_id,
                )
                _do_align_args.append((keys, dict(true=true, pred=pred)))

        if len(_do_align_args):
            print('Aligning using multiprocessing...', flush=True)
            # Create a Pool of workers
            # processes=None will default to the number of CPU cores
            pool = multiprocessing.Pool(processes=8)
            try:
                alignments = pool.starmap(_do_align, _do_align_args)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print('term', flush=True)
                pool.terminate()    
                pool.join()
                raise
        
            for row, alignment in alignments:
                self.cache.add_row(alignment, overwrite=True, **row)
        
    def _does_match_any_of_dataset_specs(
        self,
        dataset_name: str,
        aug: str,
        dataset_specs: Sequence[DatasetSpec],
    ) -> DatasetSpec | None:
        for dataset_spec in dataset_specs:
            if (
                fnmatch(
                    dataset_name,
                    dataset_spec.name_pattern,
                    flags=NEGATE | SPLIT
                )
                and (
                    dataset_spec.augmentor == aug
                    or dataset_spec.augmentor == 'all'
                )
            ):
                return dataset_spec
        return None
    
    def _get_requested_n_samples(
        self,
        preds: pl.DataFrame,
        pipeline_name: str,
        dataset_name: str,
        dataset_spec: DatasetSpec
    ) -> pl.DataFrame | None:
        import polars as pl

        if (
            dataset_spec.n_samples == 'all'
            and dataset_spec.n_samples_mode == 'up_to'
        ):
            return preds
        
        # otherwise need to possibly filter out some samples, or skip
        # (pipeline, dataset) pair
        ordered_sample_ids = self.get_ordered_sample_ids(dataset_name)
        if (
            dataset_spec.n_samples != 'all'
            and dataset_spec.n_samples_mode == 'exactly'
            and dataset_spec.n_samples > len(ordered_sample_ids)
        ):
            raise ValueError(
                f'Wrong dataset spec "{dataset_spec.to_string()}": requested'
                f' {dataset_spec.n_samples} samples'
                f', but only {len(ordered_sample_ids)} samples found'
                f' for the dataset "{dataset_name}".'
            )
        if dataset_spec.n_samples != 'all':
            # if .n_samples > len(dataset), this will to nothing
            ordered_sample_ids = ordered_sample_ids[:dataset_spec.n_samples]

        # we keep in preds_chunk only ids found in ordered_sample_ids
        preds = preds.filter(
            pl.col('sample_id').is_in(set(ordered_sample_ids))
        )

        if (
            dataset_spec.n_samples_mode == 'exactly'
            and len(preds) < len(ordered_sample_ids)
        ):
            print(
                f'{(pipeline_name, dataset_name)}'
                ' skipping due to partially missing data:'
                f'\n\tn={dataset_spec.n_samples}'
                ' first samples are requested, but found'
                f' only {len(preds)}'
                f' of {len(ordered_sample_ids)} first samples'
            )
            return None

        return preds
    
    def get_multiple_alignments(
        self,
        dataset_name: str,
        augmentor_name: str = 'none',
        parser_name: str = 'default',
        pipeline_patterns: Sequence[str] = ('*',),
    ) -> dict[int, MultipleAlignment]:
        """Compares multiple pipelines on a dataset.
    
        See Also:
            More details and examples in the user guide
            :doc:`/guide_evaluation_dashboard`.
        
        Given a list of :code:`pipeline_patterns`, searches for all keys
        in the
        :attr:`~asr_eval.bench.loader.PredictionLoader.grouped_loaded_predictions`
        that match the given pipeline, dataset, augmentor and parser.
        Since we can only compare pipelines with the same augmentor
        and parser, this provides all the results we have: pipelines,
        and their predictons on sample ids. Importantly, different
        pipelines may have different sets of sample ids: say, we run
        the first pipeline on 100 samples and the second pipeline
        only on 10 samples. Let we have pipelines P_1, ..., P_N and
        their sets of sample ids S_1, ..., S_N. The current function
        returns a dict where keys are union(S_1, ..., S_N), and for
        each sample id a
        :class:`~asr_eval.align.alignment.MultipleAlignment` is
        provided, with all pipelines that have this id predicted. In our
        example, the function returns a dict of all sample ids, and
        for 10 of them the :code:`MultipleAlignment` has 2 pipelines,
        while for the remaining 90 ids the :code:`MultipleAlignment` has
        only 1 pipeline. Further we can: 1) visualize all the
        alignments, 2) call :func:`~asr_eval.bench.evaluator.get_dataset_data`
        function that averages metrics.
        """
        
        import polars as pl
        
        rows: list[tuple[str, int, str, float]] = []
        for key, preds_chunk in self.grouped_loaded_predictions.items():
            if (
                key.dataset_name == dataset_name
                and key.augmentor == augmentor_name
                and key.parser == parser_name
                and fnmatch(
                    key.pipeline_name,
                    pipeline_patterns,
                    flags=NEGATE | SPLIT
                )
            ):
                for sample_id, pred_info in sorted(preds_chunk.items()):
                    rows.append((
                        key.pipeline_name,
                        sample_id,
                        pred_info.text,
                        pred_info.elapsed_time,
                    ))
        preds = pl.DataFrame(
            rows,
            orient='row',
            schema=['pipeline_name', 'sample_id', 'pred_text', 'elapsed_time']
        )
        
        results: dict[int, MultipleAlignment] = {}
        for sample_id, sample_df in preds.group_by('sample_id'):
            sample_id = cast(int, sample_id[0])
            
            pipeline_names: list[str] = []
            alignments: list[Alignment] = []
            elapsed_times: list[float] = []
            
            for (
                pipeline_name, _sample_id, _pred_text, elapsed_time
            ) in sample_df.iter_rows():
                pipeline_names.append(pipeline_name)
                elapsed_times.append(elapsed_time)
                alignments.append(self.cache.get_row(
                    pipeline_name=pipeline_name,
                    dataset_name=dataset_name,
                    augmentor=augmentor_name,
                    parser=parser_name,
                    sample_id=sample_id,
                    artifact_type='alignment',
                ))
            
            # sort by registering order (TODO alternative: lexicographically?)
            order = np.argsort([
                get_pipeline_index(p) for p in pipeline_names
            ]).tolist()
            pipeline_names = [pipeline_names[i] for i in order]
            alignments = [alignments[i] for i in order]
            elapsed_times = [elapsed_times[i] for i in order]
            
            annotation = self.get_annotation(
                dataset_name=dataset_name,
                parser_name=parser_name,
                sample_id=sample_id,
            )
            
            results[sample_id] = MultipleAlignment( # type: ignore
                baseline=annotation,
                alignments=dict(zip(
                    pipeline_names, alignments
                )),
                elapsed_times=dict(zip(
                    pipeline_names, elapsed_times
                )),
            )
        return results
    
    def get_ordered_sample_ids(self, dataset_name: str) -> list[int]:
        """For a given registered dataset, returns a sequence of
        sample ids in the standard (shuffled) version, that can be
        obtained by :code:`get_dataset(dataset_name, shuffle=True)`.
        """

        keys = {
            'dataset_name': dataset_name,
            'artifact_type': 'ordered_sample_ids'
        }
        try:
            return self.cache.get_row(**keys)
        except KeyError:
            dataset = get_dataset(
                dataset_name, split='test', filter=True, shuffle=True
            )
            ordered_sample_ids = cast(list[int], dataset['sample_id'])
            self.cache.add_row(ordered_sample_ids, overwrite=True, **keys)
            return ordered_sample_ids

    def get_annotation(
        self,
        dataset_name: str,
        parser_name: str | Literal['default'],
        sample_id: int
    ) -> Transcription:
        """Get a parsed annotation for the given dataset, parser name
        and sample id. If not in cache, retrieves the annotation by
        instantiating this dataset.
        """

        cache_keys = {
            'dataset_name': dataset_name,
            'parser': parser_name,
            'sample_id': sample_id,
            'artifact_type': 'annotation',
        }
        try:
            return self.cache.get_row(**cache_keys)
        except KeyError:
            transcription = _get_transcription_by_sample_id(
                dataset_name=dataset_name, sample_id=sample_id,
            )
            annotation = (
                get_parser(parser_name, 'true')
                .parse_transcription(transcription)
            )
            self.cache.add_row(annotation, overwrite=True, **cache_keys)
            return annotation
    

def _do_align(
    keys: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], Alignment]:
    """To use in parallel calls."""

    print(f'Aligning: {keys}', flush=True)
    return keys, Alignment(**kwargs)