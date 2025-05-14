import threading
import time
from typing import Self

import numpy as np

from .model import InputBuffer, RECORDING_ID_TYPE, Signal


class StreamingAudioSender:
    def __init__(
        self,
        send_to: InputBuffer,
        id: RECORDING_ID_TYPE = 0,
        item_count: int = 5,
        delay: float = 1.0,
        size: int = 16_000,
    ):
        self._buffer = send_to
        self.id = id
        self._item_count = item_count
        self._delay = delay
        self._size = size
        self._thread = None
    
    def start_sending(self) -> Self:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self
    
    def join(self) -> None:
        self._thread.join()
        self._thread = None
    
    def _run(self) -> None:
        """Produce items and add them to the buffer"""
        try:
            for i in range(self._item_count):
                chunk = (self.id, np.zeros(self._size))
                print(f'Sending: {chunk}')
                self._buffer.send(chunk)
                time.sleep(self._delay)
            
            print("Sending done")
            self._buffer.send((self.id, Signal.FINISH))
        except BaseException as e:
            self._buffer.set_error_state(e)
            raise e