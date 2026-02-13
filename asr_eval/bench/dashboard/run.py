import argparse
from collections.abc import Sequence
import importlib
from pathlib import Path
from functools import lru_cache
import re
from typing import Any, Literal
import uuid

import dash
from dash.html import Div
from dash import dcc, html, Input, Output, State, DiskcacheManager
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import diskcache

from asr_eval import CACHE_DIR
from asr_eval.bench.dashboard.header import DashboardHeader
from asr_eval.bench.datasets._registry import _add_custom_annotations_from_csv # pyright: ignore[reportPrivateUsage]
from asr_eval.bench.evaluator import (
    DatasetData, compare_pipelines, get_dataset_data
)
from asr_eval.bench.loader import PredictionLoader
from asr_eval.bench.pipelines import get_pipeline_index
from asr_eval.utils.storage import make_storage
from asr_eval.bench.dashboard.utils import (
    get_audio_assets_url,
    render_metric_plots,
    render_metric_tables,
    render_samples,
)
from asr_eval.bench.dashboard.css import (
    BOLD,
    DEFAULT_FONT,
    HEADER_STYLE,
    TAB_DESCR_STYLE,
    TAB_HEADER_STYLE,
    TAB_STYLE,
    FLEX_ROW,
    PAD,
)


__all__ = [
    'run_dashboard',
]


def run_dashboard(
    loader: PredictionLoader,
    assets_dir: str | Path = 'tmp/dashboard_assets',
    pre_export_audio: bool = False,
    host: str = '0.0.0.0',
    port: int = 8051,
):
    """Runs a web dashboard to visualize the predictions of the ASR
    models and their metrics.
    
    Has also a CLI version, see
    :code:`python -m asr_eval.bench.dashboard.run --help`
    
    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.
    
    Args:
        loader: Prediction loader that loads and aligns predictions.
        assets_dir: Directory for web assets (creates if not exists).
        pre_export_audio: Export audio .mp3 to the assets dir while
            starting the dashboard. If False, will export .mp3 on
            demand, but this may slow down the response to the user
            requests.
        host: A dashboard host.
        port: A dashboard port.
    """
    # DatasetData instances retrieved by dashboard users
    # (need to store them for pipeline pair comparison)
    session_data: dict[str, DatasetData] = {}
    
    # assets dir
    # NOTE: if use relative assets_folder, Dash threats Path(__file__).parent
    # as the working directory, not the directory from where the script was run
    # so, we need to .resolve() if before passing into dash.Dash()
    ASSETS_DIR = Path(assets_dir).resolve()
    
    # pre-exporting audio if needed
    if pre_export_audio:
        for key, preds in loader.grouped_loaded_predictions.items():
            for sample_id in preds:
                get_audio_assets_url(
                    key.dataset_name, key.augmentor, sample_id, ASSETS_DIR
                )
                
    # DISABLED: polars .group_by() hangs in a callback with this setting
    use_background_calls = False
    
    if use_background_calls:
        # a diskcache needed to support callbacks with background=True
        cache = diskcache.Cache(CACHE_DIR / 'dashboard_diskcache')
        cache.clear()
        background_callback_manager = DiskcacheManager(cache)
    else:
        background_callback_manager = None
    
    app = dash.Dash(
        __name__,
        title='ASR Dashboard',
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
        background_callback_manager=background_callback_manager,
    )

    # constructing a header with all selectors
    header = DashboardHeader(loader, default_max_samples_to_render=50)
    
    def layout() -> html.Div:
        selector_pipeline_first = dcc.Dropdown(
            id='selector-pipeline-first',
            clearable=True,
            style={'width': '250px', 'fontSize': '13px'},
        )
        selector_pipeline_second = dcc.Dropdown(
            id='selector-pipeline-second',
            clearable=True,
            style={'width': '250px', 'fontSize': '13px'},
        )
        comparison_block = html.Div([
            html.Div([
                html.Div([
                    html.Label('Pipeline 1:', style=BOLD),
                    selector_pipeline_first,
                ], style=FLEX_ROW),
                html.Div([
                    html.Label('Pipeline 2:', style=BOLD),
                    selector_pipeline_second,
                ], style=FLEX_ROW),
            ], style=HEADER_STYLE),
            html.Div(id='comparision-plot'),
        ])
        return html.Div([
            dcc.Location(id='url', refresh=False),
            dcc.Store(
                id='session-id',
                storage_type='session',
                # ensure every browser session has an ID
                data=str(uuid.uuid4()) 
            ),
            header.dash_header,
            dcc.Tabs([
                dcc.Tab(
                    label='Hide plots',
                    style=TAB_HEADER_STYLE,
                    selected_style=TAB_HEADER_STYLE,
                ),
                dcc.Tab(
                    [
                        html.P(id='tab-plots-descr', style=TAB_DESCR_STYLE),
                        Div(id='tab-plots', style=FLEX_ROW | PAD | TAB_STYLE)
                    ],
                    label='Plots',
                    style=TAB_HEADER_STYLE,
                    selected_style=TAB_HEADER_STYLE,
                ),
                dcc.Tab(
                    [
                        html.P(id='tab-tables-descr', style=TAB_DESCR_STYLE),
                        Div(id='tab-tables', style=FLEX_ROW | PAD | TAB_STYLE)
                    ],
                    label='Tables',
                    style=TAB_HEADER_STYLE,
                    selected_style=TAB_HEADER_STYLE,
                ),
                dcc.Tab(
                    Div(
                        comparison_block,
                        id='tab-compare',
                        style=FLEX_ROW | PAD | TAB_STYLE,
                    ),
                    label='Compare',
                    style=TAB_HEADER_STYLE,
                    selected_style=TAB_HEADER_STYLE,
                ),
            ], style={'margin-bottom': '5px'}),
            Div(id='multiple-alignments'),
        ], style=DEFAULT_FONT)

    app.layout = layout
    
    @lru_cache(maxsize=100)
    def get_dataset_data_cached(
        dataset_name: str,
        augmentor: str,
        parser: str,
        pipeline_patterns: tuple[str, ...],
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
        wer_averaging_mode: Literal['plain', 'concat'] = 'concat',
        exclude_samples_with_digits: bool = False,
        max_samples_to_render: int | None = None,
    ) -> DatasetData:
        multiple_alignments = loader.get_multiple_alignments(
            dataset_name=dataset_name,
            augmentor_name=augmentor,
            parser_name=parser,
            pipeline_patterns=pipeline_patterns,
        )
        dataset_data = get_dataset_data(
            multiple_alignments=multiple_alignments,
            count_absorbed_insertions=count_absorbed_insertions,
            max_consecutive_insertions=max_consecutive_insertions,
            wer_averaging_mode=wer_averaging_mode,
            exclude_samples_with_digits=exclude_samples_with_digits,
            max_samples_to_render=max_samples_to_render,
        )
        return dataset_data
        
    @lru_cache(maxsize=100)
    def handle_retrieve_click(
        dataset_display_name: str,
        pipelines: str,
        max_samples: int,
        export_audio: bool,
        exclude_with_digits: bool,
        count_absorbed: bool,
        max_insertions: int | None,
        averaging_mode: Literal['plain', 'concat'],
        session_id: str,
    ) -> Sequence[Any] | None:
        # browser session ID should always exist by the time the button is
        # clickable
        if not session_id:
            raise PreventUpdate
        
        # calculate data to display
        print('handle_retrieve_click(): get_dataset_data_cached')
        dataset_name, augmentor, parser = (
            header.all_datasets[dataset_display_name]
        )
        dataset_data = get_dataset_data_cached(
            dataset_name=dataset_name,
            augmentor=augmentor,
            parser=parser,
            # in pipeline_patterns use tuple for lru_cache
            pipeline_patterns=tuple(pipelines.split()),
            count_absorbed_insertions=count_absorbed,
            max_consecutive_insertions=max_insertions,
            wer_averaging_mode=averaging_mode,
            exclude_samples_with_digits=exclude_with_digits,
            max_samples_to_render=max_samples,
        )
        
        # store the data to enable further calls to compare pipeline pairs
        session_data[session_id] = dataset_data
        
        # # exporting to mp3 if not yet
        if export_audio:
            print('handle_retrieve_click(): export_audio')
            for sample in dataset_data.samples:
                if sample.baseline_transcription_html is not None:
                    get_audio_assets_url(
                        dataset_name,
                        augmentor,
                        sample.sample_id,
                        ASSETS_DIR,
                        do_export=True,
                    )
        
        audio_urls = {
            sample.sample_id: get_audio_assets_url(
                dataset_name,
                augmentor,
                sample.sample_id,
                ASSETS_DIR,
                do_export=False,
            )
            for sample in dataset_data.samples
        }
        
        # render all multiple alignments into a div block
        print('handle_retrieve_click(): render_samples')
        multiple_alignments_block = render_samples(
            dataset_data, audio_urls if export_audio else None
        )
        
        # render all metric plots
        print('handle_retrieve_click(): render_metric_plots')
        plots = render_metric_plots(dataset_data)
        
        # render all metric tables
        print('handle_retrieve_click(): render_metric_tables')
        tables = render_metric_tables(dataset_data)
        
        print('handle_retrieve_click(): finalizing')
        # list pipelines to return for pair comparison
        current_pipelines = sorted(
            dataset_data.get_all_pipelines(), key=get_pipeline_index
        )
        
        n_full_samples = len(dataset_data.full_samples)
        descr = [
            'Metrics obtained for: ',
            html.Span(str(n_full_samples), style=BOLD),
            ' samples (for which all pipeline predictions are available)'
        ]
        
        print('handle_retrieve_click(): done')
        return (
            multiple_alignments_block,  # multiple-alignments, children
            plots, # tab-plots, children
            descr, # tab-plots-descr, children
            tables, # tab-tables, children
            descr, # tab-tables-descr, children
            current_pipelines, # selector-pipeline-first, options
            None, # selector-pipeline-first, value
            current_pipelines, # selector-pipeline-second, options
            None, # selector-pipeline-second, value
            None, # comparision-plot, children
        )
    
    
    
    @app.callback( # type: ignore
        Output('multiple-alignments', 'children'),
        Output('tab-plots', 'children'),
        Output('tab-plots-descr', 'children'),
        Output('tab-tables', 'children'),
        Output('tab-tables-descr', 'children'),
        Output('selector-pipeline-first', 'options'),
        Output('selector-pipeline-first', 'value'),
        Output('selector-pipeline-second', 'options'),
        Output('selector-pipeline-second', 'value'),
        Output('comparision-plot', 'children', allow_duplicate=True),
        ###
        Input('apply-selectors-button', 'n_clicks'),
        ###
        State('selector-dataset', 'value'),
        State('selector-pipeline', 'value'),
        State('selector-max-samples', 'value'),
        State('selector-export-audio', 'value'),
        State('selector-exclude-with-digits', 'value'),
        State('selector-count-absorbed', 'value'),
        State('selector-max-insertions-enabled', 'value'),
        State('selector-max-insertions', 'value'),
        State('selector-averaging-mode', 'value'),
        ###
        State('session-id', 'data'),
        ###
        running=[
            (Output('apply-selectors-button', 'disabled'), True, False),
            (Output('apply-selectors-button', 'children'),
                header.button_running_text, header.button_text),
            (Output('multiple-alignments', 'children'), None, None),
        ],
        ###
        prevent_initial_call=True,
        background=use_background_calls,
    )
    def handle_retrieve_click_outer(  # pyright: ignore[reportUnusedFunction]
        n_clicks: int,
        dataset_display_name: str,
        pipelines: str,
        _max_samples: str,
        # ['True'] if checked, [] is unchecked
        _export_audio: list[str],
        # ['True'] if checked, [] is unchecked
        _exclude_with_digits: list[str],
        # ['True'] if checked, [] is unchecked
        _count_absorbed: list[str],
        # ['True'] if checked, [] is unchecked
        _max_insertions_enabled: list[str],
        _max_insertions: str,
        averaging_mode: Literal['plain', 'concat'],
        session_id: str,
    ):
        # a wrapper for args conversion and to enable lru_cache
        
        # input data conversions (checkboxes, str to int)
        max_samples = int(_max_samples)
        export_audio = 'True' in _export_audio
        exclude_with_digits = 'True' in _exclude_with_digits
        count_absorbed = 'True' in _count_absorbed
        max_insertions_enabled = 'True' in _max_insertions_enabled
        max_insertions = (
            int(_max_insertions) if max_insertions_enabled else None
        )
        
        return handle_retrieve_click(
            dataset_display_name=dataset_display_name,
            pipelines=pipelines,
            max_samples=max_samples,
            export_audio=export_audio,
            exclude_with_digits=exclude_with_digits,
            count_absorbed=count_absorbed,
            max_insertions=max_insertions,
            averaging_mode=averaging_mode,
            session_id=session_id,
        )
    
    @app.callback( # type: ignore
        Output('comparision-plot', 'children', allow_duplicate=True),
        Input('selector-pipeline-first', 'value'),
        Input('selector-pipeline-second', 'value'),
        State('session-id', 'data'),
        prevent_initial_call=True,
    )
    def handle_pair_comparison( # pyright: ignore[reportUnusedFunction]
        first_pipeline: str,
        second_pipeline: str,
        session_id: str,
    ) -> Sequence[Any] | None:
        if not first_pipeline or not second_pipeline:
            return None
        dataset_data = session_data[session_id]
        
        comparison_results = compare_pipelines(
            dataset_data=dataset_data,
            pipeline_name_1=first_pipeline,
            pipeline_name_2=second_pipeline,
        )
        fig = comparison_results.plot()
        return [dcc.Graph(
            figure=fig,
            responsive=True,
            config ={'displayModeBar': False},
            style={'height': '250px'},
        )]

    app.run(debug=False, host=host, port=port, use_reloader=False) # type: ignore


description = """A command line wrapper around
:func:`~asr_eval.bench.dashboard.run.run_dashboard` to run web dashboard
to visualize the predictions of the ASR models and their metrics.
    
See more details and examples in the user guide
:doc:`/guide_evaluation_dashboard`.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-s',
    '--storage',
    default='tmp/storage.dbm',
    help=(
        'Path of the storage file to load the results from.'
        ' Use .csv or .dbm file extension (the latter is binary'
        ' and more efficient).'
    ),
)
parser.add_argument(
    '-c',
    '--cache',
    type=str,
    required=False,
    help=(
        'Path of the ShelfStorage to cache alignments during evaluation'
        ' (creates if not exists). If not specified, disables caching.'
    ),
)
parser.add_argument(
    '--assets_dir',
    default='tmp/dashboard_assets',
    help='Directory for web assets (creates if not exists)',
)
parser.add_argument(
    '-p',
    '--pipelines',
    nargs='*',
    help='Pipelines to load from the storage (load all if not specified)',
)
parser.add_argument(
    '-d',
    '--datasets',
    nargs='*',
    help='Datasets to load from the storage (load all if not specified)',
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
    '--export-audio',
    action='store_true',
    help=(
        'Export audio .mp3 to the assets dir while starting the dashboard.'
        ' If not set, will export .mp3 on demand, but this may slow down'
        ' the response to the user requests.'
    ),
)
parser.add_argument(
    '--host',
    default='0.0.0.0',
    help='A dashboard host',
)
parser.add_argument(
    '--port',
    type=int,
    default=8051,
    help='A dashboard port',
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
    'usage: <strong>python -m asr_eval.bench.dasboard.run</strong>',
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

    storage = make_storage(args.storage)
    cache = make_storage(args.cache)
    
    run_dashboard(
        loader=PredictionLoader(
            storage=storage,
            cache=cache,
            pipelines=args.pipelines or ('*',),
            dataset_specs=args.datasets or ('*',),
        ),
        assets_dir=args.assets_dir,
        pre_export_audio=args.export_audio,
        host=args.host,
        port=args.port,
    )