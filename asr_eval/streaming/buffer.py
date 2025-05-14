import threading
from typing import TypeVar, Generic


T = TypeVar('T')

class StreamingBuffer(Generic[T]):
    def __init__(self, name: str = 'unnamed'):
        self._name = name
        self._buffer: list[T] = []
        self._lock = threading.Lock()
        self._data_available = threading.Event()
        self._error: BaseException | None = None
    
    def send(self, data: T) -> None:
        """Add data to buffer (non-blocking, thread-safe)"""
        with self._lock:
            self._validate(data)
            self._buffer.append(data)
            self._data_available.set()
    
    def receive(self) -> T:
        self._data_available.wait()  # Block until data is available
        if self._error:
            raise self._error
        with self._lock:
            assert self._buffer  # spurious wakeup
            result = self._buffer.pop(0)
            if not self._buffer:
                self._data_available.clear()
            return result
    
    def _validate(self, data: T):
        pass  # may be overridden for custom checks
    
    def set_error_state(self, error: BaseException):
        """
        On assertion violation in a thread that sends into buffer, use this to mark the buffer
        as invalid. This will raise the error on reading from the buffer.
        """
        if self._error:  # already set
            return
        self._error = RuntimeError(f'Error in the StreamingBuffer "{self._name}"')
        self._error.__cause__ = error
        self._data_available.set()
        