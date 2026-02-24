# Class InputBuffer (defined in asr_eval/streaming/model.py at lines 200-349)

class InputBuffer(asr_eval.streaming.model.ASRStreamingQueue[asr_eval.streaming.model.InputChunk]):
    """ An input buffer for
    :class:`~asr_eval.streaming.model.StreamingASR`.

    This subclass adds a
    :meth:`~asr_eval.streaming.model.InputBuffer.get_with_rechunking`
    method. If it was called at least once, a rechunking mode is enabled
    and :code:`.get()` cannot be called anymore.
    """
    ...

    @typing.override
    def get(
        self, id: asr_eval.streaming.buffer.ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[asr_eval.streaming.model.InputChunk, asr_eval.streaming.buffer.ID_TYPE]:
    ...

    def get_with_rechunking(
        self,
        size: int,
        id: asr_eval.streaming.buffer.ID_TYPE | None = None,
    ) -> tuple[asr_eval.streaming.buffer.ID_TYPE, asr_eval.streaming.model.AUDIO_CHUNK_TYPE | None, bool, float]:
        """ Internally calles ;code:`.get()` as many times as needed and
        concatenates and/or slices the results to obtain the desired
        array size.

        For example, let each input chunk contain 1000 audio frames, and
        we requested :code:`size=2400`. The :code:`.get()` will be
        called 3 times, and the last chunks will be split into two
        parts, of size 400 and 600. An array of size 2400 will be
        returned, and 600 remaining elements will be kept in the
        rechunking buffer. If then we :code:`request size=100`, the
        array of size 100 will be returned without new :code:`get()`,
        and buffer will keep 500 remaining elements, and so on.

        The retuned array can be smaller than requested only if
        :code:`Signal.FINISH` reached for the ID.

        Returns:
            1. ID (equals the :code:`id` argument if was specified,
               otherwise the first available id).
            2. Audio chunk of the desired size (or less if
               :code:`Signal.FINISH` reached).
            3. A flag if :code:`Signal.FINISH` reached.
            4. The audio end time of the last recived chunk (even if its
               part is still in the rechunking buffer). TODO maybe set
               the audio end time more correctly?
        """
        ...