from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import threading
import time
from typing import Self, override

import numpy as np

from .model import InputBuffer, RECORDING_ID_TYPE, Signal


@dataclass(kw_only=True)
class AbstractStreamingAudioSender(ABC):
    audio: np.ndarray
    send_to: InputBuffer
    id: RECORDING_ID_TYPE = 0
    sampling_rate: int = 16_000
    propagate_errors: bool = True
    verbose: bool = False

    _thread: threading.Thread | None = None

    @abstractmethod
    def get_send_times(self) -> list[float]:
        ...

    def __post_init__(self):
        assert len(self.audio), 'audio has zero length'
        assert all(np.diff(self.get_send_points()) > 0), 'chunks have zero size'

    @property
    def audio_length_sec(self) -> float:
        return len(self.audio) / self.sampling_rate

    def get_send_points(self) -> list[int]:
        points = [int(t * self.sampling_rate) for t in self.get_send_times()]
        if points[-1] == points[-2]:
            # cut a possible small ending
            points = points[:-1]
        return points
    
    def start_sending(self) -> Self:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self
    
    def join(self):
        self._thread.join()
        self._thread = None
    
    def _run(self):
        points = self.get_send_points()
        try:
            for start, end in zip(points[:-1], points[1:]):
                if self.verbose:
                    print(f'Sending: id={self.id}, start={start}, end={end}')
                self.send_to.send((self.id, self.audio[start:end]))
                time.sleep(self.real_time_interval_sec)
            self.send_to.send((self.id, Signal.FINISH))
        except BaseException as e:
            if self.propagate_errors:
                self.send_to.set_error_state(e)
            raise e
        

@dataclass(kw_only=True)
class StreamingAudioSender(AbstractStreamingAudioSender):
    real_time_interval_sec: float = 1 / 25
    speed_multiplier: float = 1.0

    @override
    def get_send_times(self) -> list[float]:
        audio_interval_sec = self.real_time_interval_sec * self.speed_multiplier
        return np.arange(
            start=0,
            stop=self.audio_length_sec + audio_interval_sec - 1e-6,
            step=audio_interval_sec,
        ).tolist()