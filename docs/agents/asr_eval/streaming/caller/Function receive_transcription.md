# Function receive_transcription (defined in asr_eval/streaming/caller.py at lines 18-30)

def receive_transcription(
    asr: asr_eval.streaming.model.StreamingASR, id: asr_eval.streaming.buffer.ID_TYPE
) -> collections.abc.Iterable[asr_eval.streaming.model.OutputChunk]:
    """Blocks and waits until the full transcription (ended with
    :code:`Signal.FINISH`) received for the given ID.
    """
    ...