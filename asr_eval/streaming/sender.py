from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
import threading
import time
from typing import Any, Literal, Self, override

import numpy as np

from asr_eval.streaming.buffer import ID_TYPE

from .model import AUDIO_CHUNK_TYPE, InputBuffer, InputChunk, Signal


@dataclass(slots=True)
class Cutoff:
    """
    Let we have an audio `waveform` and two consecutive cutoffs `[c1, c2]`:
    
    `[Cutoff(tr1, ta1, pos1), Cutoff(tr2, ta2, pos2)]`.
    
    This means that:
    1. `waveform[:c1.arr_pos]` gives an audio with length `c1.t_audio`.
    2. `waveform[:c2.arr_pos]` gives an audio with length `c2.t_audio`.
    1. `waveform[c1.arr_pos:c2.arr_pos]` should be sent at the time `c2.t_real`.
    
    The differences between `t_real` and `t_audio`:
    1. `t_real` is a real world time measured from the epoch (like time.time()), and `t_audio`
    is relative: equals 0 at the beginning of the audio.
    2. When iterating over cutoffs, `t_real` may grow faster or slower than `t_audio` which
    indicates slowed down or sped up audio stream accordingly.
    """
    t_real: float
    t_audio: float
    arr_pos: int


# @dataclass(slots=True)
# class DelayInfo:
#     was_wakeup: bool
#     wait_start_time: float
#     intended_delay: float
#     wait_end_time: float


@dataclass(kw_only=True)
class BaseStreamingAudioSender(ABC):
    """
    Can be used to send audio stream to StreamingASR.
    
    Constructor fields:
    `audio`: audio stream (int waveform, float waveform or wav bytes)
    `id`: a unique recording ID (StreamingASR require a unique ID for each recording)
    `array_len_per_sec`: array length per second in audio time scale
    `verbose`: print on each chunk sent
    `track_history`: keep a history of sent chunks in `.history` field (since each input chunk
    keeps a large array, one can use `.remove_waveforms_from_history()` after receiving full
    transcription to clear these arrays inplace)
    
    Subclasses should implement `_get_send_times()` that provide a list of cutoffs: mappings
    between audio times and real times to send. In a default implementation `StreamingAudioSender`
    both audio and real times grown monotonically.
    
    Has two ways of use:
    1. Call `.start_sending()` to start sending in a separate thread.
    2. Call `.send_all_without_delays()` to send all chunks immediately.
    
    Possible `.get_status()` values:
    - "not_started": `.start_sending()` or `.send_all_without_delays()` was not called yet.
    - "started": `.start_sending()` was called, but the thread is not finished yet.
    - "finished": either `.start_sending()` or `.send_all_without_delays()` finished.
    """
    audio: AUDIO_CHUNK_TYPE
    id: ID_TYPE = 0
    array_len_per_sec: int = 16_000
    # densify: bool = False
    verbose: bool = False
    track_history: bool = True

    history: list[InputChunk] = field(default_factory=list)
    # timings_history: list[DelayInfo] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _sent_without_delays: bool = False

    @abstractmethod
    def _get_send_times(self) -> list[Cutoff]:
        '''Get send times, in real time scale and audio time scale'''
        ...
    
    def get_send_times(self) -> list[Cutoff]:
        cutoffs = [Cutoff(0, 0, 0)] + self._get_send_times()
        if cutoffs[-1].arr_pos == cutoffs[-2].arr_pos:
            # cut a possible small ending
            cutoffs = cutoffs[:-1]
        
        assert len(cutoffs) >= 2

        assert all(np.diff([c.arr_pos for c in cutoffs]) > 0), 'at least one audio chunk has zero size'
        assert all(np.diff([c.t_real for c in cutoffs]) >= 0), 'real times should not decrease'
        assert all(np.diff([c.t_audio for c in cutoffs]) >= 0), 'audio times should not decrease'
        
        return cutoffs

    def get_status(self) -> Literal['not_started', 'started', 'finished']:
        if self._sent_without_delays:
            return 'finished'
        elif not self._thread:
            return 'not_started'
        elif not self._thread._is_stopped: # type: ignore
            return 'started'
        else:
            return 'finished'

    def __post_init__(self, *_args: Any):
        assert len(self.audio), 'audio has zero length'

    @property
    def audio_length_sec(self) -> float:
        return len(self.audio) / self.array_len_per_sec
    
    def start_sending(self, send_to: InputBuffer) -> Self:
        assert self.get_status() == 'not_started'
        self._thread = threading.Thread(target=self._run, kwargs={'send_to': send_to})
        self._thread.start()
        return self
    
    def join(self):
        assert self._thread
        self._thread.join()
    
    def _send_chunk(self, send_to: InputBuffer, cutoff1: Cutoff, cutoff2: Cutoff):
        if self.verbose:
            print(
                f'Sending: id={self.id}, real {cutoff1.t_real:.3f}..{cutoff2.t_real:.3f}'
                f', audio {cutoff2.t_audio:.3f}..{cutoff2.t_audio:.3f}'
            )
        chunk = InputChunk(
            data=self.audio[cutoff1.arr_pos:cutoff2.arr_pos],
            start_time=cutoff1.t_audio,
            end_time=cutoff2.t_audio,
        )
        send_to.put(chunk, id=self.id)
        if self.track_history:
            self.history.append(chunk)
    
    def _run(self, send_to: InputBuffer):
        try:
            cutoffs = self.get_send_times()
            
            start_time = time.time()
            densify_total_saved_time = 0.
            
            for cutoff1, cutoff2 in pairwise(cutoffs):
                wait_start_time = time.time()
                if (delay := start_time - densify_total_saved_time + cutoff2.t_real - wait_start_time) > 0:
                    time.sleep(delay)
                    # if self.densify:
                    #     with send_to.consumer_waits:
                    #         was_wakeup = send_to.consumer_waits.wait(timeout=delay)
                    #         wait_end_time = time.time()
                    #         if was_wakeup:
                    #             densify_total_saved_time += delay - (wait_end_time - wait_start_time)
                    #         self.timings_history.append(DelayInfo(was_wakeup, wait_start_time, delay, wait_end_time))
                    #         if self.verbose:
                    #             print(
                    #                 f'waited for {time.time() - wait_start_time:.3f} of {delay:.3f} sec'
                    #                 + (' (woken up early by consumer)' if was_wakeup else ' (full delay)')
                    #             )
                    # else:
                    #     time.sleep(delay)
                    #     self.timings_history.append(DelayInfo(False, wait_start_time, delay, time.time()))
                    
                self._send_chunk(send_to, cutoff1, cutoff2)
                
            send_to.put(InputChunk(data=Signal.FINISH), id=self.id)
            
        except BaseException as e:
            send_to.put_error(e)
            raise e
    
    def send_all_without_delays(self, send_to: InputBuffer) -> Self:
        assert self.get_status() == 'not_started'
        try:
            cutoffs = self.get_send_times()
            for cutoff1, cutoff2 in pairwise(cutoffs):
                self._send_chunk(send_to, cutoff1, cutoff2)
            send_to.put(InputChunk(data=Signal.FINISH), id=self.id)
        except BaseException as e:
            send_to.put_error(e)
            raise e
        self._sent_without_delays = True
        return self
    
    def remove_waveforms_from_history(self):
        for chunk in self.history:
            chunk.data = b''
    
    # def undensify(self, time: float) -> float:
    #     addition = 0.
    #     for d1, d2 in pairwise(self.timings_history):
    #         assert d1.wait_start_time < d1.wait_end_time < d2.wait_start_time < d2.wait_end_time
        
    #     for delay_info in self.timings_history:
    #         if delay_info.was_wakeup:
    #             if time < delay_info.wait_start_time:
    #                 break
    #             elif time >= delay_info.wait_end_time:
    #                 addition += delay_info.intended_delay
    #             else:
    #                 # time is inside the interval
    #                 addition += (
    #                     delay_info.intended_delay
    #                     * (time - delay_info.wait_start_time)
    #                     / (delay_info.wait_end_time - delay_info.wait_start_time)
    #                 )
    #                 break
        
    #     return time + addition
            
        

@dataclass(kw_only=True)
class StreamingAudioSender(BaseStreamingAudioSender):
    """
    A default implementation of BaseStreamingAudioSender where both audio and real send times
    grow monotonically. See BaseStreamingAudioSender docstring.
    """
    real_time_interval_sec: float = 1 / 25
    speed_multiplier: float = 1.0

    @override
    def _get_send_times(self) -> list[Cutoff]:
        audio_interval_sec = self.real_time_interval_sec * self.speed_multiplier
        audio_times = np.arange(
            start=0,
            stop=self.audio_length_sec + audio_interval_sec - 1e-6,
            step=audio_interval_sec,
        )[1:]
        return [
            Cutoff(
                t_audio / self.speed_multiplier,
                t_audio,
                int(t_audio * self.array_len_per_sec)
            )
            for t_audio in audio_times.tolist()
        ]