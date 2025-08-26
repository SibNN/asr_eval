from __future__ import annotations

import argparse
from collections.abc import Container
from pathlib import Path
from typing import Literal

import dash
from dash import dcc, Input, Output
from dash.html import Div, Label, Img
from dash.dcc import Dropdown, Checklist
from dash.development.base_component import Component
from dash_extensions import Purify
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from ..bench.datasets import get_dataset_index
from ..bench.pipelines import get_pipeline_index
from ..align.metrics import plot_dataset_metric
from .evaluator import DatasetData, Evaluator, SampleData

__all__ = [
    'run_dashboard',
]


def _display_sample_as_html(sample: SampleData) -> str:
        display_rows: list[tuple[str, ...]] = []
        if sample.baseline_is_ground_truth:
            # labeled dataset
            display_rows.append((
                'time', # elapsed_time
                'E',  # n_errors
                'R',  # n_replacements
                'D',  # n_deletions
                'I',  # n_insertions
                'True', # pipeline_name
                sample.baseline_transcription_html
            ))
            for pipeline_name, pipeline_data in sample.pipelines.items():
                display_rows.append((
                    f'{pipeline_data.elapsed_time:.2f}' if not np.isnan(pipeline_data.elapsed_time) else '?',
                    str(pipeline_data.metrics.n_errors),
                    str(pipeline_data.metrics.n_replacements),
                    str(pipeline_data.metrics.n_deletions),
                    str(pipeline_data.metrics.n_insertions),
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
    dataset_names = sorted(evaluator.list_datasets(), key=get_dataset_index)
    assert len(dataset_names)
    pipeline_names = sorted(evaluator.list_pipelines(), key=get_pipeline_index)
    assert len(pipeline_names)
    dataset_data: DatasetData | None = None
        
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    PAD = {'padding': '5px'}
    FLEXBOX_ROW = {'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}
    
    # inputs
    
    dataset_selector = Dropdown(
        id='dataset-selector',
        options=dataset_names,
        value=dataset_names[0],
        clearable=False,
        style=PAD | {'flex-grow': '1'},
    )
    pipeline_selector = Dropdown(
        id='pipeline-selector',
        options=pipeline_names,
        value=pipeline_names,
        clearable=False,
        multi=True,
        style=PAD | {'flex-grow': '1'},
    )
    count_absorbed_insertions = Checklist(
        id='count-absorbed-insertions',
        options=['Count absorbed insertions'],
        value=['Count absorbed insertions'],
        inline=True,
        style=PAD,
    )
    should_limit_insertions = Checklist(
        id='should-limit-insertions',
        options=['Max insertions'],
        value=['Max insertions'],
        inline=True,
        style=PAD,
    )
    max_insertions = dcc.Input(
        id="max-insertions",
        type="number",
        value= '4',
    )
    wer_averaging_mode = Dropdown(
        id='wer-averaging-mode',
        options=['plain', 'concat'],
        value='concat',
        clearable=False,
        style=PAD | {'flex-grow': '1'},
    )
    
    # outputs
    
    outputs_block = Div(id='outputs', style=PAD)
    
    # layout
    
    app.layout = Div([
        Div([
            Label('Dataset:', style=PAD), dataset_selector,
        ], style=FLEXBOX_ROW),
        Div([
            Label('Pipelines:', style=PAD), pipeline_selector,
        ], style=FLEXBOX_ROW),
        Div([
            Label('WER settings:', style=PAD),
            count_absorbed_insertions,
            should_limit_insertions,
            max_insertions,
            Label('Averaging mode:', style=PAD),
            wer_averaging_mode,
        ], style=FLEXBOX_ROW),
        outputs_block,
    ])
    
    
    @app.callback( # type: ignore
        Output('outputs', 'children'),
        [
            Input('dataset-selector', 'value'),
            Input('pipeline-selector', 'value'),
            Input('count-absorbed-insertions', 'value'),
            Input('should-limit-insertions', 'value'),
            Input('max-insertions', 'value'),
            Input('wer-averaging-mode', 'value'),
        ],
    )
    def display_dataset_summary(  # pyright:ignore[reportUnusedFunction]
        dataset_name: str,
        pipeline_names: list[str],
        _count_absorbed_insertions: list[str],
        _should_limit_insertions: list[str],
        max_insertions: str,
        wer_averaging_mode: Literal['plain', 'concat'],
    ) -> list[Component]:
        nonlocal dataset_data
        count_absorbed_insertions = 'Count absorbed insertions' in _count_absorbed_insertions
        should_limit_insertions = 'Max insertions' in _should_limit_insertions
        dataset_data = evaluator.get_dataset_data(
            dataset_name=dataset_name,
            pipeline_names=pipeline_names,
            count_absorbed_insertions=count_absorbed_insertions,
            max_consecutive_insertions=int(max_insertions) if should_limit_insertions else None,
            wer_averaging_mode=wer_averaging_mode,
        )
        html_blocks = [
            _display_sample_as_html(sample)
            for sample in dataset_data.samples
        ]
        
        
        multiple_alignments = Div(
            [Purify(html='<p>' + '</br></br>'.join(html_blocks) + '</p>')],
            style={
                'font-family': '"Consolas", "Ubuntu Mono", "Monaco", monospace',
                'white-space': 'pre',
            }
        )
        
        dataset_metric = dataset_data.dataset_metric
        wer_base64 = plot_dataset_metric(dataset_metric, what='wer', show=False)
        n_replacements_base64 = plot_dataset_metric(dataset_metric, what='n_replacements', show=False)
        n_insertions_base64 = plot_dataset_metric(dataset_metric, what='n_insertions', show=False)
        n_deletions_base64 = plot_dataset_metric(dataset_metric, what='n_deletions', show=False)
        
        IMG_FLEX = {'flex': '1 1 auto', 'max-width': '24.5%'}
        BASE64_HEADER = 'data:image/png;base64,'
        plots = Div([
            Img(id='wer-plot', src=BASE64_HEADER + wer_base64, style=IMG_FLEX),
            Img(id='n-replacements-plot', src=BASE64_HEADER + n_replacements_base64, style=IMG_FLEX),
            Img(id='n-insertions-plot', src=BASE64_HEADER + n_insertions_base64, style=IMG_FLEX),
            Img(id='n-deletions-plot', src=BASE64_HEADER + n_deletions_base64, style=IMG_FLEX),
        ], style=FLEXBOX_ROW | PAD)
        
        # pipeline pair comparison: inputs
        
        all_pipelines = sorted(dataset_data.get_all_pipelines(), key=get_pipeline_index)
    
        pipeline_1_selector = Dropdown(
            id='comparison-pipeline-1-selector',
            options=all_pipelines,
            value=all_pipelines[0] if len(all_pipelines) else None,
            clearable=True,
            style=PAD | {'flex-grow': '1'},
        )
    
        pipeline_2_selector = Dropdown(
            id='comparison-pipeline-2-selector',
            options=all_pipelines,
            value=all_pipelines[1] if len(all_pipelines) > 1 else None,
            clearable=True,
            style=PAD | {'flex-grow': '1'},
        )
    
        # mode_selector = Dropdown(
        #     id='comparison-mode-selector',
        #     options=['basic', 'detailed'],
        #     value='basic',
        #     clearable=False,
        #     style=PAD | {'width': '100px'},
        # )
        
        # pipeline pair comparison: outputs
        
        comparison_block = Div(id='comparison-outputs', style=PAD)
        
        return [
            plots,
            Div([
                # Label('Mode:', style=PAD),
                # mode_selector,
                Label('First pipeline to compare:', style=PAD),
                pipeline_1_selector,
                Label('Second pipeline to compare:', style=PAD),
                pipeline_2_selector,
            ], style=FLEXBOX_ROW),
            comparison_block,
            multiple_alignments
        ]
    
    
    @app.callback( # type: ignore
        Output('comparison-outputs', 'children'),
        [
            Input('comparison-pipeline-1-selector', 'value'),
            Input('comparison-pipeline-2-selector', 'value'),
            # Input('comparison-mode-selector', 'value'),
        ],
    )
    def pipeline_pair_comparison(  # pyright:ignore[reportUnusedFunction]
        pipeline_name_1: str | None,
        pipeline_name_2: str | None,
        # comparison_mode: Literal['basic', 'detailed'],
    ) -> list[Component]:
        nonlocal dataset_data
        assert dataset_data is not None
        
        if pipeline_name_1 is None or pipeline_name_2 is None:
            return []
        
        comparison_results = evaluator.compare_pipelines(
            dataset_data=dataset_data,
            pipeline_name_1=pipeline_name_1,
            pipeline_name_2=pipeline_name_2,
            # mode=comparison_mode,
        )
        fig = comparison_results.plot()
        
        return [dcc.Graph(
            figure=fig,
            responsive=True,
            config ={'displayModeBar': False},
            style={'height': '250px'},
        )]


    app.run(debug=False, host='0.0.0.0', port=8051, use_reloader=False) # type: ignore


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