# Function streaming_error_vs_latency_histogram (defined in asr_eval/streaming/plots.py at lines 168-223)

def streaming_error_vs_latency_histogram(
    evals: list[asr_eval.streaming.evaluation.StreamingEvaluationResults],
    ax: plt.Axes | None = None,
    max_latency: float = 10,
    # relative_to
):
    """Summarizes error percentage versus latency in a historgram, given
    evaluations for multiple samples.

    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    ...