import argparse
import re
from typing import Literal, cast
import dash
from dash import dcc, html, Input, Output
from dash.development.base_component import Component
from requests_cache import Path

from .evaluator import Evaluator
from ..align.data import MatchesList, Token
from ..align.multiple import multiple_transcriptions_alignment


def run_dashboard(root_dir: str | Path = 'outputs'):
    evaluator = Evaluator(root_dir=root_dir).load_results()
    
    dataset_names = list(evaluator.df['dataset_name'].unique()) # type: ignore
    pipeline_names = list(evaluator.df['pipeline_name'].unique()) # type: ignore
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
        paragraphs: list[Component] = []
        
        dataset_df = evaluator.df[
            (evaluator.df['dataset_name'] == dataset_name)
            & evaluator.df['pipeline_name'].isin(pipeline_names) # type: ignore
        ]
        dataset_df = dataset_df.sort_values('sample_idx') # type: ignore
        for sample_idx, sample_df in dataset_df.groupby('sample_idx'): # type: ignore
            sample_df = sample_df.sort_values('pipeline_name') # type: ignore
            _true_text, true_words = evaluator.get_ground_truth(dataset_name, int(sample_idx)) # type: ignore
            
            if sample_filter == 'unequal' and len(sample_df) == 2:
                words1, words2 = cast(list[list[Token]], sample_df['transcription_words'].tolist())
                text1 = ' '.join(str(w.value) for w in words1)
                text2 = ' '.join(str(w.value) for w in words2)
                if text1 == text2:
                    continue
            
            for word in true_words:
                assert isinstance(word, Token)
            true_words = cast(list[Token], true_words)
                
            alignments: dict[str, MatchesList] = {
                row['pipeline_name']: row['alignment']
                for i, row in sample_df.iterrows() # type: ignore
            }
            
            _msa_df, msa_string = multiple_transcriptions_alignment(true_words, alignments)
            msa_string = msa_string.replace(' ', '\xa0')
            header, body = msa_string.split('\n', maxsplit=1)
            # print(msa_string)
            paragraphs.append(html.P(
                [html.Span(str(sample_idx)), html.Br()]
                + [html.Span(header, style={'font-weight': 'bold'}), html.Br()]
                + string_to_spans(body)
            ))
        
        return paragraphs

    app.run(debug=False, host='0.0.0.0', port=8050, use_reloader=False) # type: ignore


def colorize_uppercase(text: str) -> list[html.Span]:
    # temporary solution to mark errors in color
    spans: list[html.Span] = []
    uppercase_words = list(re.finditer(r'[A-ZА-Я]+', text))
    
    pos = 0
    for i, word in enumerate(uppercase_words):
        if word.start() > pos:
            spans.append(html.Span(text[pos:word.start()]))
        spans.append(html.Span(word.group().lower(), style={'background-color': '#FF9C9C'}))
        pos = word.end()
        
    if pos < len(text):
        spans.append(html.Span(text[pos:]))
        
    return spans


def string_to_spans(text: str) -> list[html.Span | html.Br]:
    spans: list[html.Span | html.Br] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        spans += colorize_uppercase(line)
        if i != len(lines) - 1:
            spans.append(html.Br())
    return spans


if __name__ == '__main__':
    # example: `python -m asr_eval.bench.dashboard`
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', default='outputs', help='dir to load the results')
    args = parser.parse_args()
    
    run_dashboard(root_dir=args.root_dir)