from __future__ import annotations

import argparse
from collections.abc import Container
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
from dash.development.base_component import Component
from dash_extensions import Purify
import numpy as np
import pandas as pd

from .evaluator import Evaluator, SampleData


__all__ = [
    'run_dashboard',
]


def _display_sample_as_html(sample: SampleData) -> str:
        display_rows: list[tuple[str, ...]] = []
        if sample.baseline_is_ground_truth:
            # labeled dataset
            display_rows.append((
                'E',  # n_errors
                'R',  # n_replacements
                'D',  # n_deletions
                'I',  # n_insertions
                'time', # elapsed_time
                'True', # pipeline_name
                sample.baseline_transcription_html
            ))
            for pipeline_name, pipeline_data in sample.pipelines.items():
                display_rows.append((
                    str(pipeline_data.n_errors),
                    str(pipeline_data.n_replacements),
                    str(pipeline_data.n_deletions),
                    str(pipeline_data.n_insertions),
                    f'{pipeline_data.elapsed_time:.2f}' if not np.isnan(pipeline_data.elapsed_time) else '?',
                    pipeline_name,
                    pipeline_data.transcription_html,
                ))
        else:
            # unlabeled dataset
            display_rows.append((
                'time', # elapsed_time
                sample.baseline_name,
                sample.baseline_transcription_html
            ))
            for pipeline_name, pipeline_data in sample.pipelines.items():
                display_rows.append((
                    f'{pipeline_data.elapsed_time:.2f}',
                    pipeline_name,
                    pipeline_data.transcription_html,
                ))
            
        df = pd.DataFrame(display_rows)
        # the last column (transcription) contans <span> tags, so .to_string() shows it incorrectly
        display_lines = df.iloc[:, :-1].to_string( # type: ignore
            # col_space=5,
            index=False,
            header=False,
        ).split('\n')
        
        # we can use <br/> or \n for white-space: pre mode
        return f'{sample.sample_idx}<br/>' + '<br/>'.join(
            line + ' |' + transcription
            for line, transcription in zip(display_lines, df.iloc[:, -1]) # type: ignore
        )


def run_dashboard(
    root_dir: str | Path = 'outputs',
    cache_dir: str | Path = 'tmp/evaluator_cache',
    max_sample_idx: int | None = None,
    only_pipelines: Container[str] | None = None,
    only_datasets: Container[str] | None = None,
    exclude_pipelines: Container[str] = (),
    exclude_datasets: Container[str] = (),
):
    '''
    Runs an interactive dashboard to visualize the results of transcriber pipelines.
    
    See asr_eval/bench/README.md for details.
    '''
    evaluator = Evaluator(cache_dir=cache_dir)
    evaluator.load_results(
        root_dir=root_dir,
        max_sample_idx=max_sample_idx,
        only_pipelines=only_pipelines,
        only_datasets=only_datasets,
        exclude_pipelines=exclude_pipelines,
        exclude_datasets=exclude_datasets
    )
        
    app = dash.Dash(__name__)
    
    dataset_names = evaluator.list_datasets()
    assert len(dataset_names)
    dataset_selector = dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': name, 'value': name} for name in dataset_names],
        value=dataset_names[0],
        clearable=False,
    )
    
    pipeline_names = evaluator.list_pipelines()
    assert len(pipeline_names)
    pipeline_selector = dcc.Dropdown(
        id='pipeline-selector',
        options=[{'label': name, 'value': name} for name in pipeline_names],
        value=pipeline_names,
        clearable=False,
        multi=True,
    )
    
    selectors = html.Div([dataset_selector, pipeline_selector])
    
    text_field = html.Div(
        id='multiple-alignments',
        style={
            'font-family': '"Consolas", "Ubuntu Mono", "Monaco", monospace',
            'white-space': 'pre',
        }
    )
    
    app.layout = html.Div([selectors, text_field])
    
    
    @app.callback( # type: ignore
        Output('multiple-alignments', 'children'),
        [
            Input('dataset-selector', 'value'),
            Input('pipeline-selector', 'value'),
        ],
    )
    def display_dataset_summary(  # pyright:ignore[reportUnusedFunction]
        dataset_name: str,
        pipeline_names: list[str],
    ) -> list[Component]:
        dataset_data = evaluator.get_dataset_data(
            dataset_name=dataset_name,
            pipeline_names=pipeline_names,
        )
        html_blocks = [
            _display_sample_as_html(sample)
            for sample in dataset_data.samples
        ]
        return [Purify(html='<p>' + '</br></br>'.join(html_blocks) + '</p>')]

    app.run(debug=False, host='0.0.0.0', port=8050, use_reloader=False) # type: ignore


if __name__ == '__main__':
    # example: `python -m asr_eval.bench.dashboard`
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', default='outputs', help='dir to load the results')
    parser.add_argument('-c', '--cache_dir', default='tmp/evaluator_cache', help='cache dir for alignments')
    parser.add_argument('--exclude_pipeline', nargs='*')
    parser.add_argument('--exclude_dataset', nargs='*')
    parser.add_argument('--only_pipeline', nargs='*')
    parser.add_argument('--only_dataset', nargs='*')
    parser.add_argument('--max_sample_idx', type=int, required=False)
    args = parser.parse_args()
    
    run_dashboard(
        root_dir=args.root_dir,
        cache_dir=args.cache_dir,
        max_sample_idx=args.max_sample_idx,
        exclude_pipelines=args.exclude_pipeline or (),
        exclude_datasets=args.exclude_dataset or (),
        only_pipelines=args.only_pipeline or None,
        only_datasets=args.only_dataset or None,
    )