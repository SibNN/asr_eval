# Function get_audio_seconds_processed (defined in asr_eval/streaming/evaluation.py at lines 442-459)

def get_audio_seconds_processed(
    time: float, output_chunks: typing.Sequence[asr_eval.streaming.model.OutputChunk]
) -> float:
    """Given a full history of output chunks, and a :code:`time``, finds
    the last sent chunk with put timestamp before :code:`time` and
    returns its
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`. If
    no such chunks, returns 0.
    """
    ...