from __future__ import annotations

import time


class Timer(object):
    """
    1) To know how much time was spent and/or is left. Example:
    ```
    with Timer(timeout=2) as timer:
        print(timer.get_remaining_time())
        time.sleep(1)
        print(timer.get_remaining_time())
        time.sleep(1)
        print(timer.get_remaining_time())
        time.sleep(1)
        print(timer.get_remaining_time())
    ```
    >>> 2.0
    >>> 1.0
    >>> 0.001
    >>> TimeoutError: negative time left in Timer

    If timeout=0 in constructor, .get_remaining_time() will always be zero. This is useful when we
    want to treat 0 as "no timeout".
    ```
    """
    def __init__(self, timeout: float = 0):
        self.timeout = timeout
        self.is_active = False
    
    def __enter__(self) -> Timer:
        self.start_time = time.time()
        self.is_active = True
        return self

    def __exit__(self, type, value, traceback): # type: ignore
        self.end_time = time.time()
        self.is_active = False
    
    @property
    def elapsed_time(self) -> float:
        if self.is_active:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time
    
    def get_remaining_time(self, raise_on_negative: bool = True) -> float:
        assert self.is_active, 'use this object as context or run .start() first'
        if self.timeout == 0:
            return 0
        _remaining_time = round(self.start_time + self.timeout - time.time(), 2)
        if _remaining_time == 0:
            return 0.001
        elif _remaining_time < 0 and raise_on_negative:
            raise TimeoutError('negative time left in TimeoutClock')
        else:
            return _remaining_time