import threading
from typing import TypeVar, Generic


T = TypeVar('T')

class StreamingQueue(Generic[T]):
    """
    Similar to queue.Queue with the following differences:
    - typization
    - if something goes wrong, the procucer thread can put an error to the queue, and it will be raised
    in the consumer thread on the next get operation (see the example in test_error_propagation() in
    tests/test_buffer.py)
    """
    def __init__(self, name: str = 'unnamed'):
        self._name = name
        self._buffer: list[T] = []
        self._lock = threading.Lock()
        self._data_available = threading.Event()
        self._error: BaseException | None = None
    
    def put(self, data: T) -> None:
        """Add data to buffer (non-blocking, thread-safe)"""
        with self._lock:
            self._buffer.append(data)
            self._data_available.set()
    
    def get(self) -> T:
        self._data_available.wait()  # Block until data is available
        if self._error:
            raise self._error
        with self._lock:
            assert self._buffer  # spurious wakeup
            result = self._buffer.pop(0)
            if not self._buffer:
                self._data_available.clear()
            return result
    
    def put_error(self, error: BaseException):
        if self._error:  # already set
            return
        self._error = RuntimeError(f'Error in the StreamingBuffer "{self._name}"')
        self._error.__cause__ = error
        self._data_available.set()
        