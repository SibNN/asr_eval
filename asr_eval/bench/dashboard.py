import argparse
from collections.abc import Container
from typing import Literal
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
from dash.development.base_component import Component
from dash_extensions import Purify

from .evaluator import Evaluator


__all__ = [
    'run_dashboard',
]


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
        pref_baseline='whisper-large-v3',
        only_pipelines=only_pipelines,
        only_datasets=only_datasets,
        exclude_pipelines=exclude_pipelines,
        exclude_datasets=exclude_datasets,
    )
    
    dataset_names = evaluator.list_datasets()
    pipeline_names = evaluator.list_pipelines()
    assert len(dataset_names)
    
    app = dash.Dash(__name__)
    
    dataset_selector = dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': name, 'value': name} for name in dataset_names],
        value=dataset_names[0],
        clearable=False,
    )
    pipeline_selector = dcc.Dropdown(
        id='pipeline-selector',
        options=[{'label': name, 'value': name} for name in pipeline_names],
        value=pipeline_names,
        clearable=False,
        multi=True,
    )
    sample_filter_selector = dcc.Dropdown(
        id='sample-filter-selector',
        options=[{'label': 'all', 'value': 'all'}, {'label': 'unequal', 'value': 'unequal'}],
        value='all',
        clearable=False,
    )
    selectors = html.Div([dataset_selector, pipeline_selector, sample_filter_selector])
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
            Input('sample-filter-selector', 'value'),
        ],
    )
    def display_multiple_alignments(  # pyright:ignore[reportUnusedFunction]
        dataset_name: str,
        pipeline_names: list[str],
        sample_filter: Literal['all', 'unequal'],
    ) -> list[Component]:
        multiple_alignments = evaluator.get_multiple_alignments(dataset_name, pipeline_names)
        html_contents = ''
        for sample_idx, multiple_alignment in multiple_alignments.items():
            names = [multiple_alignment.baseline_name] + list(multiple_alignment.alignments)
            timings = [
                f'[{evaluator.get_prediction(dataset_name, sample_idx, name).elapsed_time:.2f} sec]'
                if isinstance(name, str) else ''
                for name in names
            ]
            timings = [x.ljust(13) for x in timings]
            html_contents += (
                f'{sample_idx}</br>'
                + multiple_alignment.view().render_as_text(mode='html', prefixes=timings)
                + '</br></br>'
            )
        return [Purify(html='<p>' + html_contents+ '</p>')]

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