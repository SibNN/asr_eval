import threading


__all__ = [
    'StreamingQueue',
]


ID_TYPE = int | str
"""An ID type alias for `StreamingQueue`."""


class StreamingQueue[T]:
    """Similar to :code:`queue.Queue` with the following differences:
    - Type-parameterized. This is a generic class for any element type.
    - Each element has an ID (not unique), and we can :code:`.get()` the
      next element for a specific ID. For example, IDs can be audio
      recording IDs for each audio chunk, when transcribing multiple
      recordings in parallel.
    - If an exception occurs, the producer thread can :code:`.put()` the
      exception into the queue, instead of the next chunk. It will be
      raised in the consumer thread on the next :code:`.get()`
      operation.
    """

    def __init__(self, name: str = 'unnamed'):
        self._name = name
        self._buffer: list[tuple[T, ID_TYPE]] = []
        # TODO `dict[ID_TYPE, deque[T]]` would give O(1) removal
        self._error: RuntimeError | None = None

        # should be re-entrant to support wrappers like ASRStreamingQueue
        self._condition = threading.Condition(threading.RLock())
    
    def put(self, data: T, id: ID_TYPE = 0) -> None:
        """Add new element into a queue (non-blocking, thread-safe)."""
        with self._condition:
            self._buffer.append((data, id))
            self._condition.notify_all()
    
    def _pop_element_for_id(self, id: ID_TYPE) -> tuple[T, ID_TYPE] | None:
        for i, (_item, item_id) in enumerate(self._buffer):
            if item_id == id:
                return self._buffer.pop(i)
        return None
    
    def get(
        self, id: ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[T, ID_TYPE]:
        """Wait for an element to appear in the queue, pop and return
        it (blocking, thread-safe).
        
        Args:
            id: The required ID to get. If None, will return an element
                with any ID. If not None, will return only an element
                with the specified ID.
            timeout: if set, will raise :code:`TimeoutError` if waiting
                takes longer than :code:`timeout` seconds.
        """
        with self._condition:  # acquires the lock
            while True:
                if self._error:
                    raise self._error
                if id is not None:
                    # waiting for a specific ID
                    if (popped := self._pop_element_for_id(id)) is not None:
                        return popped
                else:
                    # waiting for any ID
                    if len(self._buffer):
                        return self._buffer.pop(0)
                # if we are here, we need to wait for the next .notify_all()
                # .wait() releases the lock, waits and then re-acquires
                # the lock
                was_timeout = not self._condition.wait(timeout=timeout)
                if was_timeout:
                    raise TimeoutError
    
    def put_error(self, error: BaseException):
        """Set the queue into error state.
        
        Any consumers that try to :code:`.get()` from the queue will
        receive this exception wrapped into a :code:`RuntimeError`. This
        allows to propagate exceptions from sender to consumer thread
        to terminate the program.
        """
        with self._condition:
            if self._error:  # already set
                return
            self._error = RuntimeError(
                f'Error in the StreamingBuffer "{self._name}"'
            )
            self._error.__cause__ = error
            self._condition.notify_all()