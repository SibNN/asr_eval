import argparse
from collections.abc import Sequence
import importlib
import warnings
from typing import cast
import re

import wcmatch.fnmatch

from asr_eval.bench.pipelines import pipelines_registry
from asr_eval.bench.datasets import datasets_registry
from asr_eval.bench.datasets.dataset_spec import DatasetSpec
from asr_eval.utils.storage import make_storage, BaseStorage
from asr_eval.utils.timer import Timer
from asr_eval.bench.pipelines import TranscriberPipeline, get_pipeline
from asr_eval.bench.datasets import AudioSample, get_dataset
from asr_eval import check_datasets_package


__all__ = [
    'run_pipeline',
]


def run_pipeline(
    storage: BaseStorage,
    pipeline_name: str,
    dataset_specs: Sequence[str | DatasetSpec],
    print_transcriptions: bool = False,
    overwrite_existing: bool = False,
    suffix: str | None = None,
    keep: list[str] | None = None,
):
    '''
    Runs a pipeline on a list of datasets.
    
    Has also a CLI version, see
    :code:`python -m asr_eval.bench.run --help`
    
    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.
    
    Args:
        storage: Storage to save the results, such as
            :class:`~asr_eval.utils.storage.shelf_storage.ShelfStorage`.
        pipeline_name: Pipeline name to run.
        dataset_specs: List of dataset names, patterns or specs.
        print_transcriptions: Print transcriptions at runtime.
        overwrite_existing: Overwrite existing results, instead of
            skipping them.
        suffix: If not None, add the suffix to the pipeline name when
            saving to storage. Useful for versioning.
        keep: If not empty, keeps only the specified fields in the
            outputs of
            :class:`~asr_eval.bench.pipelines.TranscriberPipeline`.
            Can be used if the storage (e.g. .csv files) does not
            support data types for other fields.
    '''
    check_datasets_package()
    
    dataset_specs = [
        DatasetSpec.from_string(x) if isinstance(x, str) else x
        for x in dataset_specs
    ]
    
    # lazy pipeline instantiation
    pipeline_cls = get_pipeline(pipeline_name)
    pipeline_obj: TranscriberPipeline | None = None
    
    if suffix:
        pipeline_name += suffix

    for dataset_spec in dataset_specs:
        for dataset_name in list(datasets_registry):
            if wcmatch.fnmatch.fnmatch(
                dataset_name,
                dataset_spec.name_pattern,
                flags=wcmatch.fnmatch.NEGATE | wcmatch.fnmatch.SPLIT,
            ):
                print(
                    'running', pipeline_name, dataset_name, end='...\n',
                    flush=True
                )
                augmentor_name = (
                    dataset_spec.augmentor
                    if dataset_spec.augmentor != 'all'
                    else 'none'
                )
                dataset = get_dataset(
                    name=dataset_name,
                    augmentor_name=augmentor_name,
                    shuffle=True,
                )
                if (
                    dataset_spec.n_samples != 'all'
                    and len(dataset) > dataset_spec.n_samples
                ):
                    dataset = dataset.take(dataset_spec.n_samples)
                storage_keys = {
                    'pipeline_name': pipeline_name,
                    'dataset_name': dataset_name,
                    'augmentor': augmentor_name,
                }
                
                existing_samples: list[int]
                if overwrite_existing:
                    existing_samples = []
                else:
                    existing_data = storage.list_all(
                        load_values=False, **storage_keys
                    )
                    existing_samples: list[int] = (
                        existing_data['sample_id'].unique().to_list()
                        if 'sample_id' in existing_data.columns
                        else []
                    )
                
                for sample in cast(Sequence[AudioSample], dataset):
                    # try to skip sample if ready
                    if sample['sample_id'] in existing_samples:
                        continue
                    
                    # lazy pipeline instantiation
                    if pipeline_obj is None:
                        pipeline_obj = pipeline_cls(warmup=True)
                    
                    print(
                        'running',
                        str(storage_keys | {'sample_id': sample['sample_id']}),
                        end='... ',
                        flush=True
                    )
                    with Timer() as timer:
                        outputs = pipeline_obj.run(sample)
                        if print_transcriptions:
                            print(f'\t{outputs["text"]}')
                        for k, v in outputs.items():
                            if keep and k not in keep:
                                continue
                            storage.add_row(
                                value=v,
                                overwrite=overwrite_existing,
                                **storage_keys,
                                sample_id=sample['sample_id'],
                                artifact_type=k,
                            )
                    print(f'done in {timer.elapsed_time:.3f} sec', flush=True)
                
                print('done', str(storage_keys), flush=True)


description = """A command line wrapper around
:func:`~asr_eval.bench.run.run_pipeline` to run pipeline(s) on
dataset(s) and save the results.

Note that :code:`run_pipeline()` as Python function
accepts a single pipeline name, while the command line tool allows to
specify multiple names or patterns.
    
See more details and examples in the user guide
:doc:`/guide_evaluation_dashboard`.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-p',
    '--pipeline',
    nargs='+',
    required=True,
    help=(
        'A list of pipeline names or patterns. Searches for all piplines'
        ' in the registry that match patterns, then call run_pipeline()'
        ' for each of the found pipelines. For example,'
        ' "--pipeline gigaam-* whisper-tiny" will run all pipelines'
        ' starting from "gigaam-", and a "whisper-tiny" pipeline.'
    ),
    metavar='PATTERN',
)
parser.add_argument(
    '-d',
    '--dataset',
    nargs='+',
    required=True,
    help='A list of dataset names, patterns or specs.',
    metavar='PATTERN',
)
parser.add_argument(
    '-s',
    '--storage',
    default='tmp/storage.dbm',
    help=(
        'Path of the storage file to save the results'
        ' (creates if not exists).'
        ' Use .csv or .dbm file extension (the latter is binary'
        ' and more efficient).'
    ),
    metavar='PATH',
)
parser.add_argument(
    '--print',
    action='store_true',
    help='Print transcriptions to stdout.',
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Overwrite existing results, instead of skipping them.',
)
parser.add_argument(
    '--suffix',
    default=None,
    help=(
        'Add suffix to each pipeline name when saving to storage.'
        ' Useful for versioning.'
    ),
)
parser.add_argument(
    '--keep',
    nargs='+',
    help=(
        'Keep only the specified fields in the outputs of'
        ' :class:`~asr_eval.bench.pipelines.TranscriberPipeline`.'
        ' Can be used if the storage (e.g. .csv files) does not'
        ' support data types for other fields.'
    ),
)
parser.add_argument(
    '--import',
    dest='import_',
    nargs='+',
    help=(
        'Will import this module by name. Useful to register'
        'additional components, such as `my_package.asr.models`.'
    ),
)


# --- Code for building docs ---
cli_block_for_docs = re.sub(
    r'usage: .*?.py',
    'usage: <strong>python -m asr_eval.bench.run</strong>',
    parser.format_help()
)
__doc__ = (
    # need a literal block (like .. code-block::) but with word wrap
    f'{description}\n\n.. raw:: html\n\n\t'
    + '<pre style="white-space: pre-wrap">'
    + cli_block_for_docs.replace('\n', '<br>')
    + '</pre>'
)
parser.description = description
# --- End code for building docs ---


if __name__ == '__main__':
    args = parser.parse_args()
    
    storage = make_storage(args.storage)

    if args.import_:
        for statement in args.import_:
            importlib.import_module(statement)
            print(f'Imported {statement}')

    for pipeline_pattern in args.pipeline:
        was_run = False
        for pipeline_name in pipelines_registry:
            if wcmatch.fnmatch.fnmatch(
                pipeline_name,
                pipeline_pattern,
                flags=wcmatch.fnmatch.NEGATE | wcmatch.fnmatch.SPLIT,
            ):
                run_pipeline(
                    storage=storage,
                    pipeline_name=pipeline_name,
                    dataset_specs=args.dataset,
                    print_transcriptions=args.print,
                    overwrite_existing=args.overwrite,
                    suffix=args.suffix,
                    keep=args.keep,
                )
                was_run = True
        
        if not was_run:
            warnings.warn(
                f'Pipeline pattern {pipeline_pattern}'
                ' did not match any registered pipelines'
            )