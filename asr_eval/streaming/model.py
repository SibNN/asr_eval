from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import math
import threading
from typing import Any, Literal, Self, TypeVar, override

import numpy.typing as npt

from .buffer import StreamingQueue
from .transcription import PartialTranscription


class Signal(Enum):
    """See StreamingBlackBoxASR docstring for details"""
    EXIT = 0
    FINISH = 1


RECORDING_ID_TYPE = int | str
AUDIO_CHUNK_TYPE = npt.NDArray[Any] | bytes

INPUT_CHUNK_TYPE = tuple[RECORDING_ID_TYPE, AUDIO_CHUNK_TYPE | Literal[Signal.FINISH, Signal.EXIT]]
OUTPUT_CHUNK_TYPE = tuple[RECORDING_ID_TYPE, PartialTranscription | Literal[Signal.FINISH, Signal.EXIT]]

CHUNK_TYPE = TypeVar('CHUNK_TYPE', INPUT_CHUNK_TYPE, OUTPUT_CHUNK_TYPE)

class StreamingBufferWithChecks(StreamingQueue[CHUNK_TYPE]):
    """
    A StreamingBuffer to use in StreamingBlackBoxASR, with custom data consistency checks.
    """
    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._finished_ids: set[RECORDING_ID_TYPE] = set()
        self._exited = False
    
    @override
    def _validate(self, data: CHUNK_TYPE):
        try:
            id, chunk = data
            assert not self._exited, f'Got {id}, {chunk}, but already exited'
            if chunk is Signal.EXIT:
                self._exited = True
                return
            
            assert id not in self._finished_ids, f'Got {id}, {chunk}, but already finished this id'
            if chunk is Signal.FINISH:
                self._finished_ids.add(id)
        except BaseException as e:
            # this will raise the error in the receiver thread on .receive()
            self.put_error(e)
            # raise the error in the sender thread
            raise e


InputBuffer = StreamingBufferWithChecks[INPUT_CHUNK_TYPE]
OutputBuffer = StreamingBufferWithChecks[OUTPUT_CHUNK_TYPE]


class StreamingBlackBoxASR(ABC):
    """
    An abstract class that accepts a stream of input chunks and emits a stream of output chunks.
    
    Definitions:
    - **Audio chunk**: a part of a waveform. Sampling rate is defined in the class constructor. For example,
    a 10 sec mono recording with rate 16_000 can be represented as 10 chunks, each with shape (16_000,).
    Several channels can also be supported for some models. The chunk length is not restricted, models in
    general should be able to work with input chunks of any size.
    - **PartialTranscription**: a partial transcription that either add new words to the transcription or edit
    the previous words. See details in the class docstring.
    - **Recording ID**: a unique int or string identifier for a recording. This is useful if several recordings
    are streamed simultaneously, and we should know which audio recording each chunk belongs to. IDs should be
    unique for StreamingBlackBoxASR object and should not be reused, or the exception will be thrown.
    - **Signal.FINISH**: a symbol that signals that a stream for a specific recording ID has ended. This can
    refer to either the input stream (audio chunks) or the output stream (PartialTranscription-s).
    - **Signal.EXIT**: a symbol that signals that all streams for all recording IDs have ended. This can refer
    to either the input stream or the output stream. After receiving and emitting EXIT, StreamingBlackBoxASR
    thread finishes.
    
    Each input chunk can be one of:
    - a tuple (Recording ID, Audio chunk)
    - a tuple (Recording ID, Signal.FINISH)
    - a tuple (0, Signal.EXIT)
    
    Each output chunk can be one of:
    - a tuple (Recording ID, PartialTranscription)
    - a tuple (Recording ID, Signal.FINISH)
    - a tuple (0, Signal.EXIT)
    
    Details:
    - (ID, FINISH) input chunk indices that the audio for the ID has been fully sent
    - (ID, FINISH) output chunk indices that (ID, FINISH) input chunk received and the transcription is done
    - (0, EXIT) input chunk indicates that all audios have been fully sent
    - (0, EXIT) output chunk indicates that (0, EXIT) input chunk received, all transcriptions are done
    
    After creating an StreamingBlackBoxASR object, we should start a thread that will process input chunks and
    emit output chunks. After this, new audio chunks can be sent using `.input_buffer.send(...)` (non-blocking),
    and the outputs can be received with `.output_buffer.receive(...)` (blocks until output becomes available).
    Instead of manual sending, a StreamingAudioSender can be helpful. It will start a thread that sends audio
    chunks with a delay between each chunk. Example:
    
    ```
    asr = DummyBlackBoxASR(sampling_rate=16_000)
    asr.start_thread()
    ...TODO examples in progress
    ```
    
    Exception handling:
    1) Any exception raised from the StreamingBlackBoxASR thread will set the output buffer in the error state.
    This will raise the exception when reading from the output buffer.
    2) Trying to write invalid data into the input buffer (including reusing previous IDs) may set it into the
    error state. This will raise the exception when reading from the input buffer in the StreamingBlackBoxASR
    thread, then see pt. 1.
    3) Exceptions in StreamingAudioSender thread will set the input buffer into the error state, then see pt. 2.
    """
    def __init__(self, sampling_rate: int = 16_000):
        self._sampling_rate = sampling_rate
        self._thread: threading.Thread | None = None
    
    def start_thread(self) -> Self:
        """Start the processor in a background thread"""
        assert self._thread is None
        self.input_buffer = InputBuffer(name='input buffer')
        self.output_buffer = OutputBuffer(name='output buffer')
        self._thread = threading.Thread(target=self._run_and_send_exit, daemon=True)
        self._thread.start()
        return self
    
    def stop_thread(self) -> None:
        assert self._thread
        self.input_buffer.put((0, Signal.EXIT))
        self._thread.join()
        self._thread = None
    
    def _run_and_send_exit(self):
        try:
            assert self._thread and self._thread.ident == threading.get_ident()
            self._run()
        except BaseException as e:
            # catch any exception (maybe originating from input buffer being in error state, or not)
            # set the output buffer in the error state
            self.output_buffer.put_error(e)
            raise e
        self.output_buffer.put((0, Signal.EXIT))
    
    @abstractmethod
    def _run(self):
        """
        This method will be called in a separate thread on `self.start_thread()` and should live
        until Signal.EXIT has been received. To get the next input chunk, we can use
        `self.input_buffer.receive()` (blocks until the next chunk is available). To emit a new
        output chunk, we can use `self.output_buffer.send()` (non-blocking).
        
        This method should return only after Signal.EXIT chunk is received and return without
        sending EXIT chunk (this will be done in _run_and_send_exit wrapper).
        """
        ...


class DummyASR(StreamingBlackBoxASR):
    """
    Will transcribe N seconds long audio into "1 2 ... N"
    """
    
    def __init__(self, sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._received_seconds: dict[RECORDING_ID_TYPE, float] = defaultdict(float)
        self._transcribed_seconds: dict[RECORDING_ID_TYPE, int] = defaultdict(int)
    
    @override
    def _run(self):
        while True:
            id, chunk = self.input_buffer.get()
            if chunk is Signal.EXIT:
                return
            
            self._received_seconds[id] += (
                len(chunk) / self._sampling_rate if chunk is not Signal.FINISH else 0
            )
            
            new_transcribed_seconds = math.ceil(self._received_seconds[id])
            
            for i in range(self._transcribed_seconds[id], new_transcribed_seconds):
                self.output_buffer.put((id, PartialTranscription(text=str(i))))
                
            self._transcribed_seconds[id] = new_transcribed_seconds
            
            if chunk is Signal.FINISH:
                del self._received_seconds[id]
                del self._transcribed_seconds[id]
                self.output_buffer.put((id, Signal.FINISH))