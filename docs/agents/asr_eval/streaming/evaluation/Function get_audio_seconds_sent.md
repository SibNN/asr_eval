# Function get_audio_seconds_sent (defined in asr_eval/streaming/evaluation.py at lines 427-440)

def get_audio_seconds_sent(
    time: float, input_chunks: typing.Sequence[asr_eval.streaming.model.InputChunk]
) -> float:
    """Given a full history of input chunks, and a :code:`time`, finds
    the last sent chunk with put timestamp before the :code:`time` and
    returns its :code:`.end_time`. If no such chunks, returns 0.
    """
    ...