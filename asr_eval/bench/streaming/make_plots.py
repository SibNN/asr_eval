import argparse
import importlib
from pathlib import Path
import re
from typing import Literal, cast
import pickle
import json

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from asr_eval.utils.storage import make_storage
from asr_eval.bench.datasets import get_dataset_sample_by_id
from asr_eval.bench.datasets._registry import _add_custom_annotations_from_csv # pyright: ignore[reportPrivateUsage]
from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.timings import CannotFillTimings, fill_word_timings_inplace
from asr_eval.align.transcription import Transcription
from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
from asr_eval.streaming.evaluation import evaluate_streaming
from asr_eval.streaming.model import InputChunk, OutputChunk
from asr_eval.streaming.sender import Cutoff
from asr_eval.streaming.plots import (
    partial_alignments_plot,
    streaming_error_vs_latency_histogram,
    latency_plot,
    show_last_alignments,
    visualize_history,
)

description = """A command line utility for streaming evaluation.

Reads from the storage file obtained by :mod:`~asr_eval.bench.run`,
finds results of the streaming pipelines, analyzes histories of input
and output chunks and make various diagrams.

See Also:
    More details and examples in the user guide
    :doc:`/guide_evaluation_dashboard`.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-s',
    '--storage',
    default='outputs/storage.dbm',
    help=(
        'Path of the storage file to load the results from.'
        ' Use .csv or .dbm file extension (the latter is binary'
        ' and more efficient).'
    ),
)
parser.add_argument(
    '-o',
    '--output',
    default='tmp/streaming_evals',
    help='Directory to save the results',
)
parser.add_argument(
    '-a',
    '--annotations',
    nargs='*',
    help=(
        'Custom annotations for datasets not registered in asr_eval in form of'
        ' path(s) to CSV files with columns names "dataset_name", "sample_id"'
        ' and "text".'
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
    'usage: <strong>python -m asr_eval.bench.streaming.make_plots</strong>',
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

    if args.import_:
        for statement in args.import_:
            importlib.import_module(statement)
            print(f'Imported {statement}')

    if args.annotations:
        for path in args.annotations:
            _add_custom_annotations_from_csv(path)
    
    gigaam_model = GigaAMShortformCTC('v3')
    
    storage = make_storage(args.storage)
    
    OUTPUT_ROOT_DIR = Path(args.output)

    # (dataset name, sample id) -> timed transcription
    annotations_cache: dict[
        tuple[str, int],  # dataset name, sample id
        Transcription | Literal['cannot_fill']  
    ] = {}

    df = storage.list_all()
    
    # keep random order the same to debug problems
    # df = df.sort(df.hash_rows(seed=0))

    # TODO organize the code better: group by dataset, then by sample
    groups = [
        (keys, chunk_df)
        for keys, chunk_df in tqdm(df.group_by(
            ('pipeline_name', 'dataset_name', 'augmentor', 'sample_id'),
            # maintain_order=True,  # <- slows down
        ), desc='preparing groups')
        if 'input_chunks' in chunk_df['artifact_type'].unique()
        # else not a streaming pipeline
    ]
    

    # per sample evals
    for (
        group_idx,
        ((pipeline_name, dataset_name, augmentor, sample_id), chunk_df)
    ) in enumerate(groups):
        print(
            f'{group_idx}/{len(groups)}', pipeline_name,
            dataset_name, augmentor, sample_id
        )
        OUTPUT_DIR = (
            OUTPUT_ROOT_DIR / pipeline_name
            / dataset_name / augmentor / str(sample_id)
        )
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        done_flag_file = OUTPUT_DIR / 'done'
        if done_flag_file.exists():
            print('already done', flush=True)
            continue
        
        keys = dict(
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
            sample_id=sample_id,
            augmentor=augmentor,
        )
        
        # predicted
        cutoffs = cast(
            list[Cutoff],
            storage.get_row(**keys, artifact_type='cutoffs')
        )
        input_chunks = cast(
            list[InputChunk],
            storage.get_row(**keys, artifact_type='input_chunks')
        )
        output_chunks = cast(
            list[OutputChunk],
            storage.get_row(**keys, artifact_type='output_chunks')
        )
        
        # annotation and waveform
        sample = get_dataset_sample_by_id(dataset_name, 'test', sample_id)
        waveform = sample['audio']['array']
    
        
        if not (dataset_name, sample_id) in annotations_cache:
            print('filling timings...', flush=True)
            transcription = DEFAULT_PARSER.parse_transcription(
                sample['transcription']
            )
            try:
                fill_word_timings_inplace(
                    gigaam_model, waveform, transcription
                )
            except CannotFillTimings as e:
                print(repr(e), flush=True)
                transcription = 'cannot_fill'
            annotations_cache[dataset_name, sample_id] = transcription
        else:
            transcription = annotations_cache[dataset_name, sample_id]
        
        if transcription == 'cannot_fill':
            print('cannot fill timings, skipping', flush=True)
            done_flag_file.touch()
            continue
        
        print('evaluating...', flush=True)
        evaluation = evaluate_streaming(
            timed_transcription=transcription,
            waveform=waveform,
            cutoffs=cutoffs,
            input_chunks=input_chunks,
            output_chunks=output_chunks,
        )
        Path(OUTPUT_DIR / 'evaluation.pkl').write_bytes(
            pickle.dumps(evaluation)
        )
        
        plt.figure() # type: ignore
        partial_alignments_plot(evaluation)
        plt.savefig(OUTPUT_DIR / 'partial_alignments.png') # type: ignore
        plt.close()
        
        plt.figure() # type: ignore
        visualize_history(evaluation.input_chunks, evaluation.output_chunks)
        plt.savefig(OUTPUT_DIR / 'chunk_dynamics.png') # type: ignore
        plt.close()
        
        done_flag_file.touch()

    # summary plots
    for (pipeline_name, dataset_name, augmentor), chunk_df in (
        df.group_by(('pipeline_name', 'dataset_name', 'augmentor'))
    ):
        print('summaries', dataset_name, augmentor, augmentor)
        
        if 'input_chunks' not in chunk_df['artifact_type'].unique():
            continue  # not a streaming pipeline
        
        DIR = OUTPUT_ROOT_DIR / pipeline_name / dataset_name / augmentor
        
        sample_idxs = chunk_df['sample_id'].unique().to_list()
        
        done_idxs_file = DIR / 'done.json'
        if done_idxs_file.exists():
            idxs = json.loads(done_idxs_file.read_text())
            if set(idxs) == set(sample_idxs):
                print('already done', flush=True)
                continue
            print('updating stale summary plots', flush=True)
        
        evals = [
            pickle.loads(path.read_bytes())
            for sample_id in sample_idxs
            if (path := DIR / str(sample_id) / 'evaluation.pkl').exists()
        ]
        
        plt.figure() # type: ignore
        streaming_error_vs_latency_histogram(evals)
        plt.savefig(DIR / 'error_vs_latency_histogram.png') # type: ignore
        plt.close()
        
        plt.figure() # type: ignore
        latency_plot(evals)
        plt.savefig(DIR / 'latency_plot.png') # type: ignore
        plt.close()
        
        plt.figure() # type: ignore
        show_last_alignments(evals)
        plt.savefig(DIR / 'last_alignments.png') # type: ignore
        plt.close()
        
        done_idxs_file.write_text(json.dumps(sample_idxs))