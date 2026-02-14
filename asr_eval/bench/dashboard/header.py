from asr_eval.bench.loader import PredictionLoader
from asr_eval.bench.dashboard.css import (
    DROPDOWN_CONTAINER_STYLE,
    FLEX_ROW,
    BOLD,
    HEADER_STYLE,
    INPUT_STYLE,
    BUTTON_STYLE,
    WER_BUTTON_STYLE,
    WER_CHECKBOX,
    WER_CONTENT_STYLE,
)


class DashboardHeader:
    def __init__(
        self,
        loader: PredictionLoader,
        default_max_samples_to_render: int,
    ):
        from dash import dcc, html, Input, Output, State
        from dash import clientside_callback # type: ignore

        # all pipeline names to display
        self.all_pipelines = list(set([
            key.pipeline_name for key in loader.grouped_loaded_predictions
        ]))
        
        # all dataset names to display
        # a mapping (display name) -> (dataset, augmentor, normalizer)
        self.all_datasets: dict[str, tuple[str, str, str]] = {}
        for key in loader.grouped_loaded_predictions:
            joint_name = key.dataset_name
            if key.augmentor != 'none':
                joint_name += f', augmentor={key.augmentor}'
            if key.parser != 'default':
                joint_name += f', parser={key.parser}'
            self.all_datasets[joint_name] = (
                key.dataset_name, key.augmentor, key.parser
            )
            
        # creating selectors
        self.selector_dataset = dcc.Dropdown(
            id='selector-dataset',
            options=list(self.all_datasets),
            value=list(self.all_datasets)[0],
            clearable=False,
            style={'width': '300px', 'fontSize': '13px'},
        )
        self.selector_pipeline = dcc.Input(
            id='selector-pipeline',
            value='*',
            type='text',
            style=INPUT_STYLE | {'width': '100%'} 
        )
        self.selector_max_samples = dcc.Input(
            id='selector-max-samples',
            value=str(default_max_samples_to_render),
            type='number',
            style=INPUT_STYLE | {'width': '100px', 'fontSize': '13px'},
        )
        self.selector_export_audio = dcc.Checklist(
            id='selector-export-audio',
            options=[
                {'label': ' Export audio', 'value': 'True'},
            ],
            value=[],
            inline=True,
            labelStyle=FLEX_ROW | {'gap': '5px'},
        )
        self.selector_exclude_with_digits = dcc.Checklist(
            id='selector-exclude-with-digits',
            options=[
                {'label': 'Exclude samples with digits', 'value': 'True'}
            ],
            value=[],
            inline=True,
            inputStyle={'marginRight': '5px'},
            labelStyle=FLEX_ROW | {'gap': '5px'},
        )
        self.selector_count_absorbed = dcc.Checklist(
            id='selector-count-absorbed',
            options=[
                {'label': 'Count absorbed insertions', 'value': 'True'}
            ],
            value=['True'],
            inline=True,
            labelStyle=FLEX_ROW | WER_CHECKBOX,
        )
        self.selector_max_insertions_enabled = dcc.Checklist(
            id='selector-max-insertions-enabled',
            options=[
                {'label': 'Max consecutive insertions:', 'value': 'True'}
            ],
            value=['True'],
            inline=True,
            style={'marginRight': '5px'},
            labelStyle=FLEX_ROW | WER_CHECKBOX,
        )
        self.selector_max_insertions = dcc.Input(
            id='selector-max-insertions',
            value='4',
            type='number',
            style=INPUT_STYLE | {'width': '70px', 'height': '30px'}
        )
        self.selector_averaging_mode = dcc.Dropdown(
            id='selector-averaging-mode',
            options=['plain', 'concat'],
            value='concat',
            clearable=False,
            style={
                'width': '90px', 
                'fontSize': '12px',
                # Force height to 30px
                'height': '30px', 
                # Crucial: Override the default Dash min-height
                'minHeight': '30px' 
            },
        )
        self.button_text = 'Show data'
        self.button_running_text = 'Running...'
        self.apply_selectors_button = html.Button(
            self.button_text,
            id='apply-selectors-button',
            n_clicks=0,
            style=BUTTON_STYLE,
        )
        
        self.wer_settings_button = html.Button(
            [html.Span("WER Settings "), html.I(className="fa fa-cog")],
            id='wer-settings-btn',
            n_clicks=0,
            style=WER_BUTTON_STYLE,
        )
        
        self.wer_settings_content = html.Div([
            self.selector_count_absorbed,
            html.Div([
                self.selector_max_insertions_enabled,
                self.selector_max_insertions,
            ], style=FLEX_ROW | {'justifyContent': 'flex-start'}),
            html.Div([
                html.Label('Averaging mode:'),
                self.selector_averaging_mode,
            ], style=FLEX_ROW | {'justifyContent': 'flex-start'}),
        ], id='wer-settings-content', style=WER_CONTENT_STYLE)

        # 3. Assemble the Header
        self.dash_header = html.Div([
            html.Div([
                html.Label('Dataset:', style=BOLD),
                self.selector_dataset
            ], style=FLEX_ROW),
            html.Div([
                html.Label('Pipelines:', style=BOLD),
                self.selector_pipeline,
            ], style=FLEX_ROW | {'flex': '1', 'minWidth': '150px'}),
            html.Div([
                html.Label('Max samples to show:'),
                self.selector_max_samples
            ], style=FLEX_ROW),
            html.Div([
                self.selector_export_audio,
                self.selector_exclude_with_digits,
            ], style=FLEX_ROW),
            html.Div([
                self.wer_settings_button,
                self.wer_settings_content
            ], style=DROPDOWN_CONTAINER_STYLE),
            self.apply_selectors_button,
        ], style=HEADER_STYLE)

        # toggles the display style of wer_settings_content
        clientside_callback(
            """
            function(n_clicks, current_style) {
                if (!n_clicks) {return current_style;}
                
                // copy existing style to avoid overwriting other props
                var new_style = {...current_style};
                
                if (n_clicks % 2 === 1) {
                    // odd clicks: open menu
                    new_style['display'] = 'flex';
                } else {
                    // even clicks: close menu
                    new_style['display'] = 'none';
                }
                return new_style;
            }
            """,
            Output('wer-settings-content', 'style'),
            Input('wer-settings-btn', 'n_clicks'),
            State('wer-settings-content', 'style'),
            prevent_initial_call=True
        )