from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Any, Literal, Self, Sequence, TypeVar, override
import uuid

import numpy as np
import numpy.typing as npt

from .buffer import ID_TYPE, StreamingQueue


class Signal(Enum):
    """See StreamingBlackBoxASR docstring for details"""
    FINISH = 0


class Exit(Exception):
    pass


AUDIO_CHUNK_TYPE = npt.NDArray[Any] | bytes
    

@dataclass(kw_only=True)
class InputChunk:
    data: AUDIO_CHUNK_TYPE | Literal[Signal.FINISH]
    
    # chunk boundaries in the audio timescale, where 0 is the beginning of the audio
    start_time: float | None = None
    end_time: float | None = None
    
    # real-world timestamps in seconds (time.time()) filled by ASRStreamingQueue
    put_timestamp: float | None = None
    get_timestamp: float | None = None
 
    
@dataclass(kw_only=True)
class OutputChunk:
    data: TranscriptionChunk | Literal[Signal.FINISH]
    
    # total audio seconds processed, optional
    seconds_processed: float | None = None
    
    # real-world timestamps in seconds (time.time()) filled by ASRStreamingQueue
    put_timestamp: float | None = None
    get_timestamp: float | None = None


CHUNK_TYPE = TypeVar('CHUNK_TYPE', InputChunk, OutputChunk)


class ASRStreamingQueue(StreamingQueue[CHUNK_TYPE]):
    """
    A StreamingQueue to use in StreamingBlackBoxASR, with custom data consistency checks.
    """
    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._counts: dict[ID_TYPE, int] = defaultdict(int)
        self._finished_ids: set[ID_TYPE] = set()
    
    @override
    def get(self, id: ID_TYPE | None = None, timeout: float | None = None) -> tuple[CHUNK_TYPE, ID_TYPE]:
        data, id = super().get(id=id, timeout=timeout)
        data.get_timestamp = time.time()
        return data, id
    
    @override
    def put(self, data: CHUNK_TYPE, id: ID_TYPE = 0) -> None:
        self._validate(data=data, id=id)
        data.put_timestamp = time.time()
        self._counts[id] += 1
        return super().put(data, id=id)
    
    def _validate(self, data: CHUNK_TYPE, id: ID_TYPE):
        try:
            assert id not in self._finished_ids, f'Got id={id} but already finished this id'
            if data.data is Signal.FINISH:
                self._finished_ids.add(id)
        except BaseException as e:
            # this will raise the error in the receiver thread on .get()
            self.put_error(e)
            # raise the error in the sender thread
            raise e


class InputBuffer(ASRStreamingQueue[InputChunk]):
    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._rechunking_mode_on: bool = False
        self._rechunking_buffer: dict[ID_TYPE, AUDIO_CHUNK_TYPE] = {}
        self._rechunking_end_time: dict[ID_TYPE, float | None] = {}
        self._rechunking_finish_received: dict[ID_TYPE, bool] = {}
        
    @override
    def get(self, id: ID_TYPE | None = None, timeout: float | None = None) -> tuple[InputChunk, ID_TYPE]:
        assert not self._rechunking_mode_on, (
            'Method .get_with_rechunking() was called earlier. After this, you cannot call .get()'
            ' anymore because some data could have been consumed by the rechunking buffer.'
        )
        return super().get(id=id, timeout=timeout)
    
    def get_with_rechunking(
        self,
        size: int,
        id: ID_TYPE | None = None,
    ) -> tuple[ID_TYPE, AUDIO_CHUNK_TYPE | None, bool, float | None]:
        '''
        Useful if we want to rechunk input audio chunks to the desired size. Waits and returns:
        - id
        - audio chunk of the desired size or less (if finish reached)
        - flag if Signal.FINISH reached
        - audio end time in seconds
        '''
        self._rechunking_mode_on = True
        while True:
            # do we have data to return?
            if id is not None:
                if (
                    id in self._rechunking_buffer and len(self._rechunking_buffer[id]) >= size
                    or self._rechunking_finish_received.get(id, False)
                ):
                    break
            else:
                # loop over buffer and find id to return
                for _id, is_finished in self._rechunking_finish_received.items():
                    if is_finished:
                        id = _id
                        break
                for _id, buffered_chunk in self._rechunking_buffer.items():
                    if len(buffered_chunk) >= size:
                        id = _id
                        break
                
                # if id is not None now, we found an id to return
                if id is not None:
                    break
            
            # recieve next chunk from the input buffer
            chunk, recieved_id = super().get(id)
            if chunk.data is Signal.FINISH:
                self._rechunking_finish_received[recieved_id] = True
            else:
                self._rechunking_end_time[recieved_id] = chunk.end_time
                if recieved_id in self._rechunking_buffer:
                    if isinstance(chunk.data, bytes):
                        self._rechunking_buffer[recieved_id] += chunk.data
                    else:
                        # numpy array
                        self._rechunking_buffer[recieved_id] = np.concatenate([
                            self._rechunking_buffer[recieved_id], chunk.data
                        ])
                else:
                    self._rechunking_buffer[recieved_id] = chunk.data
              
        # we will return data from buffer for the given id
        assert id is not None
                
        if not id in self._rechunking_buffer:
            # no data in buffer, is_finished == True
            assert self._rechunking_finish_received[id]
            chunk_to_return = None
            is_finished = True
        elif len(self._rechunking_buffer[id]) > size:
            # more than enough data in buffer, is_finished or not
            chunk_to_return = self._rechunking_buffer[id][:size]
            self._rechunking_buffer[id] = self._rechunking_buffer[id][size:]
            is_finished = False
        else:
            # just enough or less data in buffer, is_finished or not
            chunk_to_return = self._rechunking_buffer[id]
            is_finished = self._rechunking_finish_received.get(id, False)
            del self._rechunking_buffer[id]
            
        end_time = self._rechunking_end_time.get(id, 0)
        if is_finished:
            del self._rechunking_finish_received[id]
            self._rechunking_end_time.pop(id, None)
        
        return id, chunk_to_return, is_finished, end_time

class OutputBuffer(ASRStreamingQueue[OutputChunk]):
    pass


class StreamingBlackBoxASR(ABC):
    """
    An abstract class that accepts a stream of input chunks and emits a stream of output chunks.
    
    Definitions:
    - **Audio chunk**: a part of a waveform. Sampling rate is defined in the class constructor. For example,
    a 10 sec mono recording with rate 16_000 can be represented as 10 chunks, each with shape (16_000,).
    Several channels can also be supported for some models. The chunk length is not restricted, models in
    general should be able to work with input chunks of any size.
    - **TranscriptionChunk**: a partial transcription that either add new words to the transcription or edit
    the previous words. See details in the class docstring.
    - **Recording ID**: a unique int or string identifier for a recording. This is useful if several recordings
    are streamed simultaneously, and we should know which audio recording each chunk belongs to. IDs should be
    unique for StreamingBlackBoxASR object and should not be reused, or the exception will be thrown.
    - **Signal.FINISH**: a symbol that signals that a stream for a specific recording ID has ended. This can
    refer to either the input stream (audio chunks) or the output stream (TranscriptionChunk-s).
    - **Exit**: an exception that signals that all streams for all recording IDs have ended. This can refer
    to either the input stream or the output stream. After receiving Exit from the input buffer and sending Exit
    to the output buffer, StreamingBlackBoxASR thread finishes.
    
    Each input chunk can be one of:
    - an InputChunk(id=Recording ID, data=<Audio chunk>)
    - an InputChunk(id=Recording ID, data=Signal.FINISH)
    
    Each output chunk can be one of:
    - an OutputChunk(id=Recording ID, data=<TranscriptionChunk>)
    - an OutputChunk(id=Recording ID, data=Signal.FINISH)
    
    Details:
    - FINISH input chunk indices that the audio for the ID has been fully sent
    - FINISH output chunk indices that FINISH input chunk received fhr the given ID and the transcription is done
    
    The input chunks may optionally contain audio timings. Some models may fill `.seconds_processed` field in
    `OutputChunk` - audio seconds processed (for the current recording ID) before yielding the current output chunk.
    This may be useful, because we could send 100 chunks (let it be 10 sec in total), but the model performs slow
    calculations and has already processed only 20 chunks (2 sec in total). Depending on the testing scenario
    we can treat the result as a partial transcription of the first 2 or 10 seconds of the audio signal.
    
    An Exit exception in the input buffer indicates that all audios have been fully sent
    An Exit exception in the output buffer indicates that Exit received from the input buffer and the
    StreamingBlackBoxASR thread exited. This does not mean that all transcriptions are fully done.
    
    Also, some timestamps are automatically filled:
    1. InputChunk.put_timestamp - the time when the chunk added to the StreamingBlackBoxASR.input_buffer
    2. InputChunk.get_timestamp - the time when the chunk received from the StreamingBlackBoxASR.input_buffer
    3. OutputChunk.put_timestamp - the time when the chunk added to the StreamingBlackBoxASR.output_buffer
    4. OutputChunk.get_timestamp - the time when the chunk received from the StreamingBlackBoxASR.output_buffer
    
    Pts 1, 4 happen in the caller code, and pts 2, 3 happen in the StreamingBlackBoxASR worker thread.
    
    After creating an StreamingBlackBoxASR object, we should start a thread that will process input chunks and
    emit output chunks. After this, new audio chunks can be sent using `.input_buffer.put(...)` (non-blocking),
    and the outputs can be received with `.output_buffer.get(...)` (blocks until output becomes available).
    Instead of manual sending, a StreamingAudioSender can be helpful. It will start a thread that sends audio
    chunks with a delay between each chunk.
    
    Exception handling:
    1) Any exception raised from the StreamingBlackBoxASR thread will set the output buffer in the error state.
    This will raise the exception when reading from the output buffer.
    2) Trying to write invalid data into the input buffer (including reusing previous IDs) may set it into the
    error state. This will raise the exception when reading from the input buffer in the StreamingBlackBoxASR
    thread, then see pt. 1.
    3) Exceptions in StreamingAudioSender thread will set the input buffer into the error state, then see pt. 2.
    4) `Exit` is a special exception type indicating that input or output stream has been closed properly.
    
    On how to subclass a StreamingBlackBoxASR, see _run method docstring.
    """
    def __init__(self, sampling_rate: int = 16_000):
        self._sampling_rate = sampling_rate
        self._thread: threading.Thread | None = None
    
    def start_thread(self) -> Self:
        """Start the processor in a background thread"""
        assert self._thread is None
        self.input_buffer = InputBuffer(name='input buffer')
        self.output_buffer = OutputBuffer(name='output buffer')
        self._thread = threading.Thread(target=self._run_and_send_exit)
        self._thread.start()
        return self
    
    def stop_thread(self) -> None:
        assert self._thread
        self.input_buffer.put_error(Exit())
        self._thread.join()
        self._thread = None
    
    def _run_and_send_exit(self):
        try:
            assert self._thread and self._thread.ident == threading.get_ident()
            self._run()
            # if we are here, _run has ended without Exit exception, this is not usual
            self.output_buffer.put_error(Exit())
        except BaseException as e:
            if isinstance(e, RuntimeError) and isinstance(e.__cause__, Exit):
                self.output_buffer.put_error(Exit())
                # if we are here, _run has ended after receiving Exit from the input stream, this is usual
            else:
                # if we are here, _run has ended with a different exception, maybe originating from the
                # input buffer or the problem in the _run method itself. We propagate the exception to the
                # output buffer and re-raise
                self.output_buffer.put_error(e)
                raise e
    
    @abstractmethod
    def _run(self):
        """
        This method will be called in a separate thread on `self.start_thread()` and should live forever.
        To get the next input chunk, we can use `self.input_buffer.get()` (blocks until the next chunk
        is available). To emit a new output chunk, we can use `self.output_buffer.put()` (non-blocking).
        Usually this is done in a `while True` loop.
        
        If you want to join or split input chunks to a desired size, use the following call:
        `self.input_buffer.get_with_rechunking(size: int, id: ID_TYPE | None = None)`
        
        For example, if 16_000 floats/sec are streamed, and an exteral sender sends 100ms chunks, but you
        want to get 1s chunk for any recording ID, call `self.input_buffer.get_with_rechunking(size=16_000)`.
        This will block until 10 chunks are accumulated for any ID and return the result (see
        `InputBuffer.get_with_rechunking` docstring).
        
        Normally on .stop_thread() `Exit` exception is raised from `self.input_buffer.get()` and
        should not be handled in _run.
        """
        ...


class DummyASR(StreamingBlackBoxASR):
    """
    Will transcribe N seconds long audio into "1 2 ... N"
    """
    
    def __init__(self, sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._received_seconds: dict[ID_TYPE, float] = defaultdict(float)
        self._transcribed_seconds: dict[ID_TYPE, int] = defaultdict(int)
    
    @override
    def _run(self):
        while True:
            chunk, id = self.input_buffer.get()
            
            self._received_seconds[id] += (
                len(chunk.data) / self._sampling_rate if chunk.data is not Signal.FINISH else 0
            )
            
            new_transcribed_seconds = math.ceil(self._received_seconds[id])
            
            for i in range(self._transcribed_seconds[id], new_transcribed_seconds):
                self.output_buffer.put(OutputChunk(data=TranscriptionChunk(text=str(i))), id=id)
                
            self._transcribed_seconds[id] = new_transcribed_seconds
            
            if chunk.data is Signal.FINISH:
                del self._received_seconds[id]
                del self._transcribed_seconds[id]
                self.output_buffer.put(OutputChunk(data=Signal.FINISH), id=id)


def new_uid() -> str:
    return str(uuid.uuid4())


@dataclass(kw_only=True)
class TranscriptionChunk:
    """
    A chunk returned by a streaming ASR model, may contain any text and any UID. If the model
    wants to edit the previous chunk, it can yield the same UID with another text. Example:
    
    TranscriptionChunk.join([
        TranscriptionChunk(text='a'),               # append a new chunk with text 'a' without an explicit uid to refer
        TranscriptionChunk(uid=1, text='b'),         # append a new chunk with text 'b', 2 chunks in total: 'a', 'b'[uid=1]
        TranscriptionChunk(uid=2, text='c'),         # append a new chunk with text 'c', 3 chunks in total: 'a', 'b'[uid=1], 'c'[uid=2]
        TranscriptionChunk(uid=1, text='b2 b3'),     # edit the chunk with uid=1: 'a', 'b2 b3'[uid=1], 'c'[uid=2]
    ]) == 'a b2 b3 c'
    """
    uid: int | str = field(default_factory=new_uid)
    text: str
    
    @classmethod
    def join(
        cls,
        transcriptions: Sequence[TranscriptionChunk | OutputChunk | Literal[Signal.FINISH]],
    ) -> str:
        parts: dict[int | str, str] = {}
        
        for t in transcriptions:
            if isinstance(t, OutputChunk):
                t = t.data
            if t is Signal.FINISH:
                continue
            parts[t.uid] = t.text
            
        return ' '.join(parts.values())