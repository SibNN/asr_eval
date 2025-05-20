from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import time
from typing import Any, Self, override

import numpy as np

from .model import AUDIO_CHUNK_TYPE, InputBuffer, RECORDING_ID_TYPE, InputChunk, Signal


@dataclass(kw_only=True)
class BaseStreamingAudioSender(ABC):
    """
    Can be used to send int waveform, float waveform or wav bytes
    """
    audio: AUDIO_CHUNK_TYPE
    id: RECORDING_ID_TYPE = 0
    sampling_rate: int = 16_000
    propagate_errors: bool = True
    verbose: bool = False

    _thread: threading.Thread | None = None

    @abstractmethod
    def get_send_times(self) -> list[tuple[float, float]]:
        '''Get send times, in real time scale and audio time scale'''
        ...

    def __post_init__(self, *_args: Any):
        assert len(self.audio), 'audio has zero length'

    @property
    def audio_length_sec(self) -> float:
        return len(self.audio) / self.sampling_rate
    
    def start_sending(self, send_to: InputBuffer) -> Self:
        self._thread = threading.Thread(target=self._run, kwargs={'send_to': send_to}, daemon=True)
        self._thread.start()
        return self
    
    def join(self):
        assert self._thread
        self._thread.join()
        self._thread = None
    
    def _run(self, send_to: InputBuffer):
        try:
            times = self.get_send_times()
            points = [int(t_audio * self.sampling_rate) for _t_real, t_audio in times]
            if points[-1] == points[-2]:
                # cut a possible small ending
                times = times[:-1]
                points = points[:-1]

            assert all(np.diff(points) > 0), 'at least one audio chunk has zero size'
            assert all(np.diff([t_real for t_real, _t_audio in times]) >= 0), 'real times should not decrease'
            assert all(np.diff([t_audio for _t_real, t_audio in times]) >= 0), 'audio times should not decrease'

            for i, (
                (tr1, ta1),  # real start time, audio start time
                (tr2, ta2),  # real end time, audio end time
                p1, p2       # audio start and end points
            ) in enumerate(zip(times[:-1], times[1:], points[:-1], points[1:])):
                if self.verbose:
                    print(f'Sending: id={self.id}, real {tr1:.3f}..{tr2:.3f}, audio {ta1:.3f}..{ta2:.3f}')
                send_to.put(InputChunk(id=self.id, data=self.audio[p1:p2]))
                if i != len(times) - 2:  # don't sleep after the last chunk
                    time.sleep(tr2 - tr1)
            send_to.put(InputChunk(id=self.id, data=Signal.FINISH))
        except BaseException as e:
            if self.propagate_errors:
                send_to.put_error(e)
            raise e
        

@dataclass(kw_only=True)
class StreamingAudioSender(BaseStreamingAudioSender):
    real_time_interval_sec: float = 1 / 25
    speed_multiplier: float = 1.0

    @override
    def get_send_times(self) -> list[tuple[float, float]]:
        audio_interval_sec = self.real_time_interval_sec * self.speed_multiplier
        audio_times = np.arange(
            start=0,
            stop=self.audio_length_sec + audio_interval_sec - 1e-6,
            step=audio_interval_sec,
        )
        real_times = audio_times / self.speed_multiplier
        return list(zip(real_times.tolist(), audio_times.tolist()))