from dataclasses import dataclass, field
from itertools import pairwise
import threading
import time
from typing import Literal, Self

import numpy as np

from asr_eval.utils.misc import new_uid
from asr_eval.utils.types import FLOATS
from asr_eval.utils.audio_ops import convert_audio_format
from asr_eval.streaming.buffer import ID_TYPE
from asr_eval.streaming.model import InputBuffer, InputChunk, Signal, StreamingASR


__all__ = [
    'Cutoff',
    'get_uniform_cutoffs',
    'StreamingSender',
]


@dataclass(slots=True)
class Cutoff:
    """A container for audio position and a real time (relative) moment.
    Is used to shedule a waveform sending into
    :class:`~asr_eval.streaming.model.StreamingASR`.
    
    Let we have an audio `waveform` and two consecutive cutoffs
    :code:`[c1, c2]`:

    .. code-block:: python
    
        [Cutoff(tr1, ta1, pos1), Cutoff(tr2, ta2, pos2)]
    
    This means that:

    1. :code:`waveform[:c1.arr_pos]` gives an audio with length
       :code:`c1.t_audio`.
    2. :code:`waveform[:c2.arr_pos]` gives an audio with length
       :code:`c2.t_audio`.
    1. :code:`waveform[c1.arr_pos:c2.arr_pos]` should be sent at the
       time :code:`c2.t_real`.
    """
    t_real: float
    """A real world time measured from the beginning of the sending
    process.
    """

    t_audio: float
    """A time in the audio timescale. For example, if we send 10 sec
    audio in 5 seconds (with 2x speed), the :code:`t_real` will be 5 in
    the end, and :code:`t_audio` will be 10.
    """

    arr_pos: int
    """A position in the audio as array. Is calculated from the
    :code:`t_audio` using array length per second.
    """


def get_uniform_cutoffs(
    waveform: FLOATS,
    real_time_interval_sec: float = 1 / 25,
    speed_multiplier: float = 1.0,
    sampling_rate: int = 16_000,
) -> list[Cutoff]:
    """Returns a uniform shedule to send the audio.

    Args:
        waveform: The audio in float32 dtype.
        real_time_interval_sec: How often in real time to send chunks?
        speed_multiplier: For example, if :code:`speed_multiplier=2`,
            will sent the audio twice of normal speed, that is, a 10
            seconds audio will be sent in 5 seconds.
        sampling_rate: The sampling rate of the :code:`waveform`.
    """
    audio_length_sec = len(waveform) / sampling_rate
    audio_interval_sec = (
        real_time_interval_sec * speed_multiplier
    )
    audio_times = np.arange(
        0,
        stop=audio_length_sec + audio_interval_sec - 1e-6,
        step=audio_interval_sec,
    )[1:]
    return [Cutoff(0, 0, 0)] + [
        Cutoff(
            t_audio / speed_multiplier,
            t_audio,
            int(t_audio * sampling_rate)
        )
        for t_audio in audio_times.tolist()
    ]


def _validate_cutoffs(cutoffs: list[Cutoff]) -> list[Cutoff]:
    if cutoffs[-1].arr_pos == cutoffs[-2].arr_pos:
        # cut a possible small ending
        cutoffs = cutoffs[:-1]
    
    assert len(cutoffs) >= 2

    assert all(np.diff([c.arr_pos for c in cutoffs]) > 0), (
        'at least one audio chunk has zero size'
    )
    assert all(np.diff([c.t_real for c in cutoffs]) >= 0), (
        'real times should not decrease'
    )
    assert all(np.diff([c.t_audio for c in cutoffs]) >= 0), (
        'audio times should not decrease'
    )

    return cutoffs


@dataclass
class StreamingSender:
    """ Can be used to automate sending audio stream to
    :class:`~asr_eval.streaming.model.StreamingASR`.

    Args:
        cutoffs: A schedule to send the audio.
        waveform: A waveform in float32 dtype. The sampling rate
            information is encoded in cutoffs, because they store both
            the audio time and the audio array position.
        asr: A streaming transcriber to send chunks into.
        id: A recording ID to assign.
        verbose: Whether to print each chunk info to stdout.
    
    Call
    :meth:`~asr_eval.streaming.sender.StreamingSender.start_sending()`
    to start a sending process.

    Note:
        Keeps the history of all sent chunks for evaluation purposes
        (can be retrieved with
        :meth:`~asr_eval.streaming.sender.StreamingSender.join_and_get_history`).
        To avoid out of memory, ensure that senders are
        garbage-collected afterwards.
    """

    cutoffs: list[Cutoff]
    waveform: FLOATS
    asr: StreamingASR
    id: ID_TYPE = field(default_factory=new_uid)
    verbose: bool = False
    sampling_rate: int = 16_000

    _history: list[InputChunk] = field(default_factory=list[InputChunk])
    _thread: threading.Thread | None = None
    _done = False

    def __post_init__(self):
        self.cutoffs = _validate_cutoffs(self.cutoffs)

    @property
    def audio_length_sec(self) -> float:
        return len(self.waveform) / self.sampling_rate

    def start_sending(self, without_delays: bool = False) -> Self:
        """If :code:`without_delays=False` (default) starts sending in a
        separate thread according to the shedule given in constuctor.
        If :code:`without_delays=True` sends all the chunks immediately.
        Non-blocking.
        """
            
        assert self.get_status() == 'not_started'
        if without_delays:
            self._send_all_without_delays(send_to=self.asr.input_buffer)
        else:
            self._thread = threading.Thread(
                target=self._run, kwargs={'send_to': self.asr.input_buffer}
            )
            self._thread.start()
        return self
    
    def join(self):
        """Wait for the sending process to finish."""

        match self.get_status():
            case 'not_started':
                raise RuntimeError(
                    'Cannot .join() when sending was not started'
                )
            case 'started':
                assert self._thread is not None
                self._thread.join()
            case 'finished':
                pass
        
    
    def join_and_get_history(self) -> list[InputChunk]:
        """Wait for the sending process to finish and return the history
        of chunks sent.
        """

        self.join()
        return self._history.copy()
    
    def _send_chunk(
        self, send_to: InputBuffer, cutoff1: Cutoff, cutoff2: Cutoff
    ):
        if self.verbose:
            print(
                f'Sending: id={self.id}'
                f', real {cutoff1.t_real:.3f}..{cutoff2.t_real:.3f}'
                f', audio {cutoff2.t_audio:.3f}..{cutoff2.t_audio:.3f}'
            )
        data = self.waveform[cutoff1.arr_pos:cutoff2.arr_pos]
        data = convert_audio_format(data, to_audio_type=self.asr.audio_type)
        chunk = InputChunk(
            data=data,
            end_time=min(cutoff2.t_audio, self.audio_length_sec),
        )
        send_to.put(chunk, id=self.id)
        self._history.append(chunk)
    
    def _run(self, send_to: InputBuffer):
        try:
            start_time = time.time()
            densify_total_saved_time = 0.
            
            for cutoff1, cutoff2 in pairwise(self.cutoffs):
                wait_start_time = time.time()
                delay = (
                    start_time
                    - densify_total_saved_time
                    + cutoff2.t_real
                    - wait_start_time
                )
                if delay > 0:
                    time.sleep(delay)
                    
                self._send_chunk(send_to, cutoff1, cutoff2)
                
            send_to.put(InputChunk(
                data=Signal.FINISH,
                end_time=min(self.cutoffs[-1].t_audio, self.audio_length_sec),
            ), id=self.id)
            self._done = True
            
        except BaseException as e:
            send_to.put_error(e)
            raise e
    
    def _send_all_without_delays(self, send_to: InputBuffer):
        try:
            for cutoff1, cutoff2 in pairwise(self.cutoffs):
                self._send_chunk(send_to, cutoff1, cutoff2)
            send_to.put(InputChunk(
                data=Signal.FINISH,
                end_time=min(self.cutoffs[-1].t_audio, self.audio_length_sec),
            ), id=self.id)
        except BaseException as e:
            send_to.put_error(e)
            raise e
        self._done = True

    def get_status(self) -> Literal['not_started', 'started', 'finished']:
        """ Possible statuses:

        - not_started: Sending was not started.
        - started: Sending in progress.
        - finished: Sending finished.
        """

        if self._done:
            return 'finished'
        elif not self._thread:
            return 'not_started'
        else:
            return 'started'