import argparse
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


def run_dashboard(root_dir: str | Path = 'outputs'):
    '''
    Runs an interactive dashboard to visualize the results of transcriber pipelines.
    
    See asr_eval/bench/README.md for details.
    '''
    evaluator = Evaluator(root_dir=root_dir).load_results(pref_baseline='whisper-large-v3', dataset_names=['podlodka-full'])
    
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
        multiple_alignments = {
            sample_idx: multiple_alignment.get_names(pipeline_names)
            for (_dataset_name, sample_idx), multiple_alignment
            in evaluator.multiple_alignments.items()
            if _dataset_name == dataset_name
        }
        
        # sort by sample idx
        multiple_alignments = dict(sorted(multiple_alignments.items()))
        
        html_contents = ''
        for sample_idx, multiple_alignment in multiple_alignments.items():
            html_contents += (
                f'{sample_idx}</br>'
                + multiple_alignment.view().render_as_text(mode='html')
                + '</br></br>'
            )
        return [Purify(html='<p>' + html_contents+ '</p>')]

    app.run(debug=False, host='0.0.0.0', port=8050, use_reloader=False) # type: ignore


if __name__ == '__main__':
    # example: `python -m asr_eval.bench.dashboard`
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', default='outputs', help='dir to load the results')
    args = parser.parse_args()
    
    run_dashboard(root_dir=args.root_dir)