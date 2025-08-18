from __future__ import annotations

import argparse
from collections.abc import Container
from dataclasses import dataclass
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
from dash.development.base_component import Component
from dash_extensions import Purify
import numpy as np
import pandas as pd

from .loader import PredictionLoader


__all__ = [
    'run_dashboard',
]


@dataclass
class DatasetData:
    samples: list[SampleData]


@dataclass
class SampleData:
    sample_idx: int
    baseline_transcription_html: str
    baseline_is_ground_truth: bool
    pipelines: dict[str, SamplePipelineData]
    baseline_name: str = ''


@dataclass
class SamplePipelineData:
    n_errors: int
    n_replacements: int
    n_insertions: int
    n_deletions: int
    elapsed_time: float
    transcription_html: str


class EvaluatorDataModel:
    def __init__(
        self,
        root_dir: str | Path = 'outputs',
        cache_dir: str | Path = 'tmp/evaluator_cache',
        max_sample_idx: int | None = None,
        only_pipelines: Container[str] | None = None,
        only_datasets: Container[str] | None = None,
        exclude_pipelines: Container[str] = (),
        exclude_datasets: Container[str] = (),
    ):
        self._loader = PredictionLoader(cache_dir=cache_dir)
        self._loader.load_results(
            root_dir=root_dir,
            max_sample_idx=max_sample_idx,
            pref_baseline='whisper-large-v3',
            only_pipelines=only_pipelines,
            only_datasets=only_datasets,
            exclude_pipelines=exclude_pipelines,
            exclude_datasets=exclude_datasets,
        )
    
    def list_datasets(self) -> list[str]:
        return self._loader.list_datasets()
    
    def list_pipelines(self) -> list[str]:
        return self._loader.list_pipelines()
    
    def get_dataset_data(
        self,
        dataset_name: str,
        pipeline_names: list[str],
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
    ) -> DatasetData:
        multiple_alignments = self._loader.get_multiple_alignments(dataset_name, pipeline_names)
        
        samples: list[SampleData] = []
        for sample_idx, multiple_alignment in multiple_alignments.items():
            baseline_is_ground_truth = multiple_alignment.baseline_name is True
            aligned_html = (
                multiple_alignment
                .view()
                .render_as_text(mode='html', html_add_style=False, add_pipeline_names=False)
                .split('<br/>')
            )
            
            pipelines: dict[str, SamplePipelineData] = {}
            for (pipeline_name, alignment), aligned_transcription in zip(
                multiple_alignment.alignments.items(), aligned_html[1:]
            ):
                pred = self._loader.get_prediction(dataset_name, sample_idx, pipeline_name)
                sample_metrics = alignment.metric_summary(
                    count_absorbed_insertions=count_absorbed_insertions,
                    max_consecutive_insertions=max_consecutive_insertions
                )
                pipelines[pipeline_name] = SamplePipelineData(
                    transcription_html=aligned_transcription,
                    elapsed_time=pred.elapsed_time,
                    n_errors=sample_metrics.n_errors,
                    n_replacements=sample_metrics.n_replacements,
                    n_insertions=sample_metrics.n_insertions,
                    n_deletions=sample_metrics.n_deletions,
                )
            
            samples.append(SampleData(
                sample_idx=sample_idx,
                baseline_transcription_html=aligned_html[0],
                baseline_is_ground_truth=baseline_is_ground_truth,
                pipelines=pipelines,
                baseline_name=str(multiple_alignment.baseline_name),
            ))
        
        return DatasetData(samples=samples)


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
    data_model = EvaluatorDataModel(
        root_dir=root_dir,
        cache_dir=cache_dir,
        max_sample_idx=max_sample_idx,
        only_pipelines=only_pipelines,
        only_datasets=only_datasets,
        exclude_pipelines=exclude_pipelines,
        exclude_datasets=exclude_datasets
    )
        
    app = dash.Dash(__name__)
    
    dataset_names = data_model.list_datasets()
    assert len(dataset_names)
    dataset_selector = dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': name, 'value': name} for name in dataset_names],
        value=dataset_names[0],
        clearable=False,
    )
    
    pipeline_names = data_model.list_pipelines()
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
    
    def display_sample_as_html(sample: SampleData) -> str:
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
        dataset_data = data_model.get_dataset_data(
            dataset_name=dataset_name,
            pipeline_names=pipeline_names,
        )
        html_blocks = [
            display_sample_as_html(sample)
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