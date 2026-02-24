# Function dataset_metric_to_dataframe (defined in asr_eval/align/metrics.py at lines 226-251)

def dataset_metric_to_dataframe(
    metrics: dict[str, asr_eval.align.metrics.DatasetMetric],
    what: typing.Literal[
        'wer', 'n_replacements', 'n_insertions', 'n_deletions'
    ] = 'wer',
    quantile_1: float = 0.1,
    quantile_2: float = 0.9,
) -> pd.DataFrame:
    """Given bootstrap distributions for several models, summarizes
    them into a table.

    A helper function for a dashboard.
    """
    ...