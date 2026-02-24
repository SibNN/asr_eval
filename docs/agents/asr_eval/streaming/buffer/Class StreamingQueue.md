# Class StreamingQueue (defined in asr_eval/streaming/buffer.py at lines 13-93)

class StreamingQueue[T]:
    """Similar to :code:`queue.Queue` with the following differences:
    - Typization. This is a generic class for any element type.
    - Each element has an ID (not unqiue), and we can :code:`.get()` the
      next element for a specific ID. For example, IDs can be audio
      recording IDs for each audio chunk, when transcribing multiple
      recordings in parallel.
    - If an exception occurs, the procucer thread can :code:`.put()` the
      exception into the queue, instead of the next chunk. It will be
      raised in the consumer thread on the next :code:`.get()`
      operation.
    """
    ...

    def put(self, data: T, id: asr_eval.streaming.buffer.ID_TYPE = 0) -> None:
        """Add new element into a queue (non-blocking, thread-safe)."""
        ...

    def get(
        self, id: asr_eval.streaming.buffer.ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[T, asr_eval.streaming.buffer.ID_TYPE]:
        """Wait for an alement to appear in the queue, pop and return
        it (blocking, thread-safe).

        Args:
            id: The required ID to get. If None, will return an element
                with any ID. If not None, will return only an element
                with the specified ID.
            timeout: if set, will raise :code:`TimeoutError` if waiting
                takes longer than :code:`timeout` seconds.
        """
        ...

    def put_error(self, error: BaseException):
        """Set the queue into error state.

        Any consumers that try to :code:`.get()` from the queue will
        recieve this exception wrapped into a :code:`RuntimeError`. This
        allows to propagate exceptions from sender to consumer thread
        to terminate the program.
        """
        ...