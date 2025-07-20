from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Any, Literal, Self, Sequence, TypeVar, override

import numpy as np
import numpy.typing as npt

from .buffer import ID_TYPE, StreamingQueue
from ..utils.misc import new_uid
from ..utils.types import FLOATS, INTS


class Signal(Enum):
    """Signals to control StreamingASR thread. See StreamingASR docstring for details."""
    FINISH = 0


class Exit(Exception):
    """A signal to terminate StreamingASR thread. See StreamingASR docstring for details."""
    pass


# Audio stream that is chunkable using slices.
AUDIO_CHUNK_TYPE = FLOATS | INTS | bytes | str
    

@dataclass(kw_only=True)
class InputChunk:
    """
    An input chunk for StreamingASR. Input chunks can be sent by BaseStreamingAudioSender or
    manually and received by StreamingASR via ASRStreamingQueue.
    
    See StreamingASR docstring for usage details.
    
    `data` is either a slice of audio, or a Signal.FINISH.
    
    `end_time` is a chunk end time (in seconds) in the audio timescale, where 0 means the beginning
    of the audio.
    
    `put_timestamp` is filled automatically when the chunk is added to the ASRStreamingQueue input buffer.
    `get_timestamp` is filled automatically when the StreamingASR thread takes the chunk from the buffer.
    """
    data: AUDIO_CHUNK_TYPE | Literal[Signal.FINISH]
    
    end_time: float
    
    put_timestamp: float = np.nan
    get_timestamp: float = np.nan
 
    
@dataclass(kw_only=True)
class OutputChunk:
    """
    An output chunk for StreamingASR. Output chunks are sent by StreamingASR and received manually or
    by `receive_full_transcription()`.
    
    See StreamingASR and `receive_full_transcription` docstring for usage details.
    
    `data` is either a part of transcription, or a Signal.FINISH.
    
    `seconds_processed` is a total audio seconds processed before yielding the current chunk.
    
    `put_timestamp` is filled automatically when the chunk is added to the ASRStreamingQueue output buffer.
    `get_timestamp` is filled automatically when the chunk is taken from the buffer.
    """
    data: TranscriptionChunk | Literal[Signal.FINISH]
    
    seconds_processed: float
    
    put_timestamp: float = np.nan
    get_timestamp: float = np.nan
    
    
def check_consistency(input_chunks: list[InputChunk], output_chunks: list[OutputChunk]):
    """
    Asserts that:
    1. For each input and output chunk put_timestamp <= get_timestamp.
    2. For each output chunk `seconds_processed` is not larger than audio seconds taken from the input
    buffer by the time the output is put into the buffer.
    
    Fails indicate errors in the chunk processing pipeline (sender, buffer or model).
    """
    for input_chunk in input_chunks:
        assert input_chunk.put_timestamp <= input_chunk.get_timestamp
    for input_chunk in output_chunks:
        assert input_chunk.put_timestamp <= input_chunk.get_timestamp
    for output_chunk in output_chunks:
        if output_chunk.data is not Signal.FINISH:
            seconds_consumed = max([
                c.end_time
                for c in input_chunks
                if c.get_timestamp < output_chunk.put_timestamp
            ])
            assert seconds_consumed >= output_chunk.seconds_processed


CHUNK_TYPE = TypeVar('CHUNK_TYPE', InputChunk, OutputChunk)


class ASRStreamingQueue(StreamingQueue[CHUNK_TYPE]):
    """
    A StreamingQueue to use in StreamingASR. It fills `put_timestamp`, `get_timestamp` and asserts
    that if Signal.FINISH was received, no more chunks are expected for this recording ID.
    
    See StreamingASR docstring for details.
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
    """
    An input buffer for StreamingASR. Input chunks are added to the buffer by `.put()` and
    received in StreamingASR thread by `.get()` or `.get_with_rechunking()`.
    
    If `.get_with_rechunking()` was called at least once, a rechunking mode is enabled and
    `.get()` cannot be called anymore.
    """
    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._rechunking_mode_on: bool = False
        self._rechunking_buffer: dict[ID_TYPE, AUDIO_CHUNK_TYPE] = {}
        self._rechunking_end_time: dict[ID_TYPE, float] = {}
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
    ) -> tuple[ID_TYPE, AUDIO_CHUNK_TYPE | None, bool, float]:
        '''
        Internally calles `.get()` as many times as needed and concatenates and/or slices the results
        to obtain the desired array size.
        
        For example, let each input chunk contain 1000 audio frames, and we requested size=2400. The `.get()`
        will be called 3 times, and the last chunks will be split into two parts, of size 400 and 600. An array
        of size 2400 will be returned, and 600 remaining elements will be kept in the rechunking buffer. If
        then we request size=100, the array of size 100 will be returned without new `get()`s, and buffer
        will keep 500 remaining elements, and so on.
        
        The retuned array can be smaller than requested if Signal.FINISH reached for the ID.
        
        Returns a tuple:
        1. id (equals the `id` argument if was specified, or the first available id otherwise).
        2. audio chunk of the desired size or less (if finish reached).
        3. flag if Signal.FINISH reached.
        4. the audio end time of the last recived chunk (even if its part is still in the rechunking buffer).
        
        TODO maybe set the audio end time more correctly?
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
                    if isinstance(chunk.data, (bytes, str)):
                        self._rechunking_buffer[recieved_id] += chunk.data # type: ignore
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
    """
    An output buffer for StreamingASR. Output chunks are added to the buffer by `.put()` in
    StreamingASR thread  and received by `.get()`.
    """
    pass


class StreamingASR(ABC):
    """
    An abstract class that accepts a stream of input chunks and emits a stream of output chunks.
    
    **Definitions:**
    
    - **Audio chunk**: a part of a waveform. Sampling rate is defined in the class constructor. For example,
    a 10 sec mono recording with rate 16_000 can be represented as 10 chunks, each with shape (16_000,).
    Several channels can also be supported for some models. The chunk length is not restricted, models in
    general should be able to work with input chunks of any size. NOTE: each StreamingASR implementation has
    `.audio_type` field that is one of {float, int, bytes}, and `.sampling_rate` field.
    - **TranscriptionChunk**: a partial transcription that either add new words to the transcription or edit
    the previous words. See details in the class docstring.
    - **Recording ID**: a unique int or string identifier for a recording. This is useful if several recordings
    are streamed simultaneously, and we should know which audio recording each chunk belongs to. IDs should be
    unique for StreamingASR object and should not be reused, or the exception will be thrown.
    - **Signal.FINISH**: a symbol that signals that a stream for a specific recording ID has ended. This can
    refer to either the input stream (audio chunks) or the output stream (TranscriptionChunk-s).
    - **Exit**: an exception that signals that all streams for all recording IDs have ended. This can refer
    to either the input stream or the output stream. After receiving Exit from the input buffer and sending Exit
    to the output buffer, StreamingASR thread finishes.
    
    **Data model:**
    
    Each input chunk can be one of:
    1. An InputChunk(id=Recording ID, data=<Audio chunk>).
    2. An InputChunk(id=Recording ID, data=Signal.FINISH) - indicates that the audio for the ID has been
    fully sent.
    
    Each output chunk can be one of:
    1. An OutputChunk(id=Recording ID, data=<TranscriptionChunk>).
    2. An OutputChunk(id=Recording ID, data=Signal.FINISH) - indicates that FINISH input chunk received fhr the
    given ID and the transcription is done.
    
    The input chunks may optionally contain audio timings. Some models may fill `.seconds_processed` field in
    `OutputChunk` - audio seconds processed (for the current recording ID) before yielding the current output chunk.
    This may be useful, because we could send 100 chunks (let it be 10 sec in total), but the model performs slow
    calculations and has already processed only 20 chunks (2 sec in total). Depending on the testing scenario
    we can treat the result as a partial transcription of the first 2 or 10 seconds of the audio signal.
    
    **Sending and receiving:**
    
    After creating an StreamingASR object, we should start a thread that will process input chunks and
    emit output chunks. After this, new audio chunks can be sent using `.input_buffer.put(...)` (non-blocking),
    and the outputs can be received with `.output_buffer.get(...)` (blocks until output becomes available).
    Instead of manual sending, a StreamingAudioSender can be helpful. It will start a thread that sends audio
    chunks with a delay between each chunk.
    
    Input and output buffers automatically fill the follwing fields:
    1. InputChunk.put_timestamp - the time when the chunk added to the StreamingASR.input_buffer
    2. InputChunk.get_timestamp - the time when the chunk received from the StreamingASR.input_buffer
    3. OutputChunk.put_timestamp - the time when the chunk added to the StreamingASR.output_buffer
    4. OutputChunk.get_timestamp - the time when the chunk received from the StreamingASR.output_buffer
    
    Pts 1, 4 happen in the caller code, and pts 2, 3 happen in the StreamingASR worker thread.
    
    **Terminating a StreamingASR thread:**
    
    An Exit exception in the input buffer indicates that all audios have been fully sent
    An Exit exception in the output buffer indicates that Exit received from the input buffer and the
    StreamingASR thread exited. This does not mean that all transcriptions are fully done.
    
    **Exception handling:**
    
    1) Any exception raised from the StreamingASR thread will set the output buffer in the error state.
    This will raise the exception when reading from the output buffer.
    2) Trying to write invalid data into the input buffer (including reusing previous IDs) may set it into the
    error state. This will raise the exception when reading from the input buffer in the StreamingASR
    thread, then see pt. 1.
    3) Exceptions in StreamingAudioSender thread will set the input buffer into the error state, then see pt. 2.
    4) `Exit` is a special exception type indicating that input or output stream has been closed properly.
    
    **Implementing models:**
    
    To subclass a StreamingASR, one should implement `._run()` and `.audio_type()` methods (see docstrings).
    """
    def __init__(self, sampling_rate: int = 16_000):
        self.sampling_rate = sampling_rate
        self._thread: threading.Thread | None = None
    
    def start_thread(self) -> Self:
        """Start _run() in a background thread"""
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
    
    @property
    @abstractmethod
    def audio_type(self) -> Literal['float', 'int', 'bytes']:
        """Returns a tuple: the first is one of {float, int, bytes}, the second is sampling rate"""
        ...



class DummyASR(StreamingASR):
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
                len(chunk.data) / self.sampling_rate if chunk.data is not Signal.FINISH else 0
            )
            
            new_transcribed_seconds = math.ceil(self._received_seconds[id])
            
            for i in range(self._transcribed_seconds[id], new_transcribed_seconds):
                self.output_buffer.put(OutputChunk(
                    data=TranscriptionChunk(text=str(i)),
                    seconds_processed=chunk.end_time,
                ), id=id)
                
            self._transcribed_seconds[id] = new_transcribed_seconds
            
            if chunk.data is Signal.FINISH:
                del self._received_seconds[id]
                del self._transcribed_seconds[id]
                self.output_buffer.put(OutputChunk(
                    data=Signal.FINISH,
                    seconds_processed=chunk.end_time,
                ), id=id)
    
    @property
    @override
    def audio_type(self) -> Literal['float', 'int', 'bytes']:
        return 'float'


@dataclass(kw_only=True)
class TranscriptionChunk:
    """
    A chunk returned by a streaming ASR model, may contain any text and any UID. If the model
    wants to edit the previous chunk, it can yield the same UID with another text. Example:
    
    TranscriptionChunk.join([
        # append a new chunk with text 'a' without an explicit uid to refer
        TranscriptionChunk(text='a'),
        
        # append a new chunk with text 'b', 2 chunks in total: 'a', 'b'[uid=1]
        TranscriptionChunk(uid=1, text='b'),
        
        # append a new chunk with text 'c', 3 chunks in total: 'a', 'b'[uid=1], 'c'[uid=2]
        TranscriptionChunk(uid=2, text='c'),
        
        # edit the chunk with uid=1: 'a', 'b2 b3'[uid=1], 'c'[uid=2]
        TranscriptionChunk(uid=1, text='b2 b3'),
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


def prepare_audio_format(
    waveform: npt.NDArray[np.floating[Any]],
    asr: StreamingASR,
    sampling_rate: int = 16_000,
) -> tuple[AUDIO_CHUNK_TYPE, int]:
    '''
    Based on asr.audio_type and asr.sampling_rate, returns:
    - the audio data to send into asr via BaseStreamingAudioSender
    - the value array_len_per_sec to specify for BaseStreamingAudioSender
    '''
    assert sampling_rate == asr.sampling_rate  # todo implement resampling
    match asr.audio_type:
        case 'float':
            audio = waveform
            array_len_per_sec = asr.sampling_rate
        case 'int':
            audio = (waveform * 32768).astype(np.int16)
            array_len_per_sec = asr.sampling_rate
        case 'bytes':
            audio = (waveform * 32768).astype(np.int16).tobytes()
            array_len_per_sec = asr.sampling_rate * 2  # x2 because of the conversion int16 -> bytes
    
    return audio, array_len_per_sec