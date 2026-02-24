# Class ASRStreamingQueue (defined in asr_eval/streaming/model.py at lines 153-198)

class ASRStreamingQueue(asr_eval.streaming.buffer.StreamingQueue[asr_eval.streaming.model.CHUNK_TYPE]):
    """An input or output buffer in
    :class:`~asr_eval.streaming.model.StreamingASR`.

    This subclass extends the
    :class:`~asr_eval.streaming.buffer.StreamingQueue`:

    1. It fills :code:`put_timestamp`, :code:`get_timestamp` for input
    or output chunks.
    2. It asserts that if :code:`Signal.FINISH` was received, no more
    chunks are expected for this audio recording ID.
    """
    ...

    @typing.override
    def get(
        self, id: asr_eval.streaming.buffer.ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[asr_eval.streaming.model.CHUNK_TYPE, asr_eval.streaming.buffer.ID_TYPE]:
    ...

    @typing.override
    def put(self, data: asr_eval.streaming.model.CHUNK_TYPE, id: asr_eval.streaming.buffer.ID_TYPE = 0) -> None:
    ...