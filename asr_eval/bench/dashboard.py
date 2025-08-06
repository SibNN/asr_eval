from typing import cast
import dash
from dash import dcc, html, Input, Output
from dash.development.base_component import Component

from .evaluator import Evaluator
from ..align.data import MatchesList, Token
from ..align.multiple import multiple_transcriptions_alignment


def run_dashboard():
    evaluator = Evaluator(root_dir='tmp').load_results()
    
    dataset_names = list(evaluator.df['dataset_name'].unique()) # type: ignore
    assert len(dataset_names)
    
    app = dash.Dash(__name__)
    
    dropdown = dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': name, 'value': name} for name in dataset_names],
        value=dataset_names[0],
        clearable=False,
    )
    text_field = html.Div(
        id='multiple-alignments',
        style={
            'font-family': '"Consolas", "Ubuntu Mono", "Monaco", monospace',
            'white-space': 'pre',
        }
    )
    app.layout = html.Div([dropdown, text_field])
    
    @app.callback( # type: ignore
        Output('multiple-alignments', 'children'),
        [Input('dataset-selector', 'value')]
    )
    def display_multiple_alignments(dataset_name: str) -> list[Component]:  # pyright:ignore[reportUnusedFunction]
        paragraphs: list[Component] = []
        
        dataset_df = evaluator.df[evaluator.df['dataset_name'] == dataset_name]
        dataset_df = dataset_df.sort_values('sample_idx') # type: ignore
        for sample_idx, sample_df in dataset_df.groupby('sample_idx'): # type: ignore
            sample_df = sample_df.sort_values('pipeline_name') # type: ignore
            _true_text, true_words = evaluator.get_ground_truth(dataset_name, int(sample_idx)) # type: ignore
            
            for word in true_words:
                assert isinstance(word, Token)
            true_words = cast(list[Token], true_words)
                
            alignments: dict[str, MatchesList] = {
                row['pipeline_name']: row['alignment']
                for i, row in sample_df.iterrows() # type: ignore
            }
            
            _msa_df, msa_string = multiple_transcriptions_alignment(true_words, alignments)
            msa_string = msa_string.replace(' ', '\xa0')
            print(msa_string)
            paragraphs.append(string_to_paragraph(f'{sample_idx}\n{msa_string}'))
        
        return paragraphs

    app.run(debug=False, host='0.0.0.0', port=8050, use_reloader=False) # type: ignore


def string_to_paragraph(text: str) -> html.P:
    spans: list[Component] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        spans.append(html.Span(line))
        if i != len(lines) - 1:
            spans.append(html.Br())
    return html.P(spans)