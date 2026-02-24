# Function check_consistency (defined in asr_eval/streaming/model.py at lines 117-147)

def check_consistency(
    input_chunks: list[asr_eval.streaming.model.InputChunk],
    output_chunks: list[asr_eval.streaming.model.OutputChunk],
):
    """ Asserts that:

    1. For each input and output chunk
       :code:`put_timestamp <= get_timestamp`.
    2. For each output chunk :code:`seconds_processed` is not larger
    than audio seconds taken from the input buffer by the time the
    output is put into the buffer.

    Fails indicate errors in the chunk processing pipeline (sender,
    buffer or model).

    Raises:
        AssertionError: On check failures.
    """
    ...