# Function visualize_history (defined in asr_eval/streaming/plots.py at lines 125-165)

def visualize_history(
    input_chunks: list[asr_eval.streaming.model.InputChunk],
    output_chunks: list[asr_eval.streaming.model.OutputChunk] | None = None,
    ax: plt.Axes | None = None,
):
    """Visualize the history of sending and receiving chunks.

    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    ...