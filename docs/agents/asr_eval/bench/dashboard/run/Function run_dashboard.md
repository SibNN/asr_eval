# Function run_dashboard (defined in asr_eval/bench/dashboard/run.py at lines 42-431)

def run_dashboard(
    loader: asr_eval.bench.loader.PredictionLoader,
    assets_dir: str | pathlib.Path = 'tmp/dashboard_assets',
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
    ...