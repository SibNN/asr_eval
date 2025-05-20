import threading
from typing import TypeVar, Generic


ID_TYPE = int | str

T = TypeVar('T')

class StreamingQueue(Generic[T]):
    """
    Similar to queue.Queue with the following differences:
    - typization
    - elements are associated with IDs (not unqiue), and we can .get() a specific ID
    - if something goes wrong, the procucer thread can put an error to the queue, and it will be raised
    in the consumer thread on the next get operation
    """
    def __init__(self, name: str = 'unnamed'):
        self._name = name
        self._buffer: list[tuple[T, ID_TYPE]] = []
        self._error: RuntimeError | None = None
        self._condition = threading.Condition()
        self.history: list[tuple[T, ID_TYPE]] | None = None
    
    def keep_history(self):
        self.history = []
    
    def put(self, data: T, id: ID_TYPE = 0) -> None:
        """Add data to buffer (non-blocking, thread-safe)"""
        with self._condition:
            self._buffer.append((data, id))
            if self.history is not None:
                self.history.append((data, id))
            self._condition.notify_all()
    
    def _pop_element_for_id(self, id: ID_TYPE) -> tuple[T, ID_TYPE] | None:
        for i, (_item, item_id) in enumerate(self._buffer):
            if item_id == id:
                return self._buffer.pop(i)
        else:
            return None
    
    def get(self, id: ID_TYPE | None = None, timeout: float | None = None) -> tuple[T, ID_TYPE]:
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
                # .wait() releases the lock, waits and then re-acquires the lock
                was_timeout = not self._condition.wait(timeout=timeout)
                if was_timeout:
                    raise TimeoutError()
    
    def put_error(self, error: BaseException):
        if self._error:  # already set
            return
        with self._condition:
            self._error = RuntimeError(f'Error in the StreamingBuffer "{self._name}"')
            self._error.__cause__ = error
            self._condition.notify_all()