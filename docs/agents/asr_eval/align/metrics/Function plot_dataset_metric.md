# Function plot_dataset_metric (defined in asr_eval/align/metrics.py at lines 253-306)

def plot_dataset_metric(
    metrics: dict[str, asr_eval.align.metrics.DatasetMetric],
    what: typing.Literal[
        'wer', 'n_replacements', 'n_insertions', 'n_deletions'
    ] = 'wer',
    show: bool = True,
    quantile_1: float = 0.1,
    quantile_2: float = 0.9,
) -> str:
    """Given bootstrap distributions for several models, summarizes
    them into a plot.

    A helper function for a dashboard.

    If :code:`show=True`, calls :code:`plt.show()` afterwards. Returns
    base64-encoded image.
    """
    ...