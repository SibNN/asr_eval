import argparse
import os
from pathlib import Path
import re
from typing import Any


description = """A command line utility for streaming evaluation.

Scans the directory created by :mod:`~asr_eval.bench.streaming.make_plots`
tool and runs a web interface to visualize the results.

See more details and examples in the user guide
:doc:`/guide_evaluation_dashboard`.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-d',
    '--dir',
    default='tmp/streaming_evals',
    help='Directory with plots, created by `make_plots` tool.',
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


# --- Code for building docs ---
cli_block_for_docs = re.sub(
    r'usage: .*?.py',
    'usage: <strong>python -m asr_eval.bench.streaming.show_plots</strong>',
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
    # LLM code

    import dash
    import flask
    from dash import dcc, html, Input, Output

    args = parser.parse_args()
    
    ROOT_DIR = Path(args.dir)
    assert ROOT_DIR.is_dir()

    server = flask.Flask(__name__)
    app = dash.Dash(__name__, server=server)
    app.title = "ASR Streaming dashboard"

    @server.route('/static_images/<path:path>')
    def serve_static(path: str | Path):
        print((ROOT_DIR / path).exists())
        try:
            response = flask.send_from_directory(str(ROOT_DIR.resolve()), str(path))
        except Exception as e:
            print(e)
            raise
        return response

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label("Select Pipeline:"),
                dcc.Dropdown(id='pipeline-dropdown', placeholder="Select Pipeline"),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '10px'}),

            html.Div([
                html.Label("Select Dataset:"),
                dcc.Dropdown(id='dataset-dropdown', placeholder="Select Dataset"),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '10px'}),

            html.Div([
                html.Label("Select Sample ID:"),
                dcc.Dropdown(id='sample-dropdown', placeholder="Select Sample"),
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'padding': '5px', 'backgroundColor': '#f9f9f9', 'borderBottom': '1px solid #ddd'}),

        html.H3("Global Metrics (Pipeline/Dataset)", style={'margin': '5px 5px 5px 20px'}),
        html.Div([
            # Plot 1
            html.Div([
                html.H4("Error vs Latency", style={'margin': '5px 5px 5px 20px'}),
                html.Div(
                    html.Img(id='img-error-latency', style={'width': '100%', 'border': '1px solid #ccc'}),
                    style={'overflow': 'hidden'}
                )
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.H4("Last Alignments", style={'margin': '5px 5px 5px 20px'}),
                html.Div(
                    html.Img(id='img-last-alignments', style={'width': '100%', 'border': '1px solid #ccc'}),
                    style={'overflow': 'hidden'}
                )
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.H4("Latency Plot", style={'margin': '5px 5px 5px 20px'}),
                html.Div(
                    html.Img(id='img-latency-plot', style={'width': '100%', 'border': '1px solid #ccc'}),
                    style={'overflow': 'hidden'}
                )
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'display': 'flex', 'justifyContent': 'center'}),

        html.Hr(),

        html.H3("Sample Specific Dynamics", style={'margin': '5px 5px 5px 20px'}),
        html.Div([
            html.Div([
                html.H4("Chunk Dynamics", style={'margin': '5px 5px 5px 20px'}),
                html.Div(
                    html.Img(id='img-chunk-dynamics', style={'width': '100%', 'border': '1px solid #ccc'}),
                    style={'overflow': 'hidden'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            html.Div([
                html.H4("Partial Alignments", style={'margin': '5px 5px 5px 20px'}),
                html.Div(
                    html.Img(id='img-partial-alignments', style={'width': '100%', 'border': '1px solid #ccc'}),
                    style={'overflow': 'hidden'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '50px'}),
    ], style={'fontFamily': 'Segoe UI, sans-serif'})

    # update pipelines
    @app.callback( # type: ignore
        Output('pipeline-dropdown', 'options'),
        Input('pipeline-dropdown', 'search_value') # Dummy input just to fire on load
    )
    def load_pipelines(_) -> list[dict[str, Any]]:
        items = [o for o in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, o))]
        return [{'label': i, 'value': i} for i in sorted(items)]

    # update datasets based on pipeline
    @app.callback( # type: ignore
        Output('dataset-dropdown', 'options'),
        Input('pipeline-dropdown', 'value')
    )
    def update_datasets(pipeline: str) -> list[dict[str, Any]]:
        if not pipeline:
            return []
        
        path = os.path.join(ROOT_DIR, pipeline)
        if not os.path.exists(path):
            return []
        
        items = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        return [{'label': i, 'value': i} for i in sorted(items)]

    # 3. Update Samples based on Pipeline and Dataset
    @app.callback( # type: ignore
        Output('sample-dropdown', 'options'),
        [Input('pipeline-dropdown', 'value'),
        Input('dataset-dropdown', 'value')]
    )
    def update_samples(pipeline: str, dataset: str) -> list[dict[str, Any]]:
        if not pipeline or not dataset:
            return []
        # Structure: root/pipeline/dataset/none/{sample_id}
        path = os.path.join(ROOT_DIR, pipeline, dataset, 'none')
        if not os.path.exists(path):
            return []
        # We look for directories inside 'none', ignoring the png files at this level
        items = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        return [{'label': i, 'value': i} for i in sorted(items)]

    @app.callback( # type: ignore
        [Output('img-error-latency', 'src'),
        Output('img-last-alignments', 'src'),
        Output('img-latency-plot', 'src'),
        Output('img-chunk-dynamics', 'src'),
        Output('img-partial-alignments', 'src')],
        [Input('pipeline-dropdown', 'value'),
        Input('dataset-dropdown', 'value'),
        Input('sample-dropdown', 'value')]
    )
    def update_graphs(pipeline: str, dataset: str, sample: str):
        # Default placeholder or empty image
        empty_src = "" 
        
        if not pipeline or not dataset:
            return empty_src, empty_src, empty_src, empty_src, empty_src
        
        # Base URL for static file serving
        base_url = "/static_images"
        
        # 1. Construct Aggregate Paths (Root/Pipeline/Dataset/none/file.png)
        # URL format: /static_images/Pipeline/Dataset/none/filename.png
        agg_path_rel = f"{pipeline}/{dataset}/none"
        
        src_hist = f"{base_url}/{agg_path_rel}/error_vs_latency_histogram.png"
        src_last = f"{base_url}/{agg_path_rel}/last_alignments.png"
        src_lat  = f"{base_url}/{agg_path_rel}/latency_plot.png"
        
        src_chunk = empty_src
        src_part  = empty_src
        
        # 2. Construct Sample Path (Root/Pipeline/Dataset/none/Sample/file.png)
        if sample:
            sample_path_rel = f"{pipeline}/{dataset}/none/{sample}"
            src_chunk = f"{base_url}/{sample_path_rel}/chunk_dynamics.png"
            src_part  = f"{base_url}/{sample_path_rel}/partial_alignments.png"
            
        return src_hist, src_last, src_lat, src_chunk, src_part

    app.run(debug=False, host=args.host, port=args.port) # type: ignore