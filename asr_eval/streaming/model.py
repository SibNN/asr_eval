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
    
    # an input chunk counter for the current recording ID, filled by ASRStreamingQueue
    index: int = -1
 
    
@dataclass(kw_only=True)
class OutputChunk:
    data: TranscriptionChunk | Literal[Signal.FINISH]
    
    # number of processed input chunks, optional
    n_input_chunks_processed: int | None = None
    
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
        # self.history: list[tuple[CHUNK_TYPE, ID_TYPE]] | None = None
        # self._positions_in_history: dict[int, int] = {}
    
    # def track_history(self):
    #     self.history = []
    
    @override
    def get(self, id: ID_TYPE | None = None, timeout: float | None = None) -> tuple[CHUNK_TYPE, ID_TYPE]:
        data, id = super().get(id=id, timeout=timeout)
        data.get_timestamp = time.time()
        # if self.history is not None and builtins.id(data) in self._positions_in_history:
        #     history_idx = self._positions_in_history[builtins.id(data)]
        #     self.history[history_idx][0].get_timestamp = data.get_timestamp
        return data, id
    
    @override
    def put(self, data: CHUNK_TYPE, id: ID_TYPE = 0) -> None:
        self._validate(data=data, id=id)
        data.put_timestamp = time.time()
        if isinstance(data, InputChunk):
            data.index = self._counts[id]
        self._counts[id] += 1
        # if self.history is not None:
        #     self.history.append((copy.deepcopy(data), id))
        #     # self.history.append((data_copy := copy.deepcopy(data), id))
        #     # if isinstance(data_copy, AUDIO_CHUNK_TYPE):  # ??
        #     #     del data_copy.data
        #     # self._positions_in_history[builtins.id(data)] = len(self.history) - 1
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


InputBuffer = ASRStreamingQueue[InputChunk]
OutputBuffer = ASRStreamingQueue[OutputChunk]


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
    
    Optionally, some models may fill `.n_input_chunks_processed` field in `OutputChunk` - a total number of input
    chunks processed (for the current recording ID) before yielding the current output chunk. This may be useful,
    because we could send 100 chunks (let it be 10 sec in total), but the model performs slow calculations and has
    already processed only 20 chunks (2 sec in total). Depending on the testing scenario, we can treat the result
    as a partial transcription of the first 2 or 10 seconds of the audio signal.
    
    An Exit exception in the input buffer indicates that all audios have been fully sent
    An Exit exception in the output buffer indicates that Exit received from the input buffer and the
    StreamingBlackBoxASR thread exited. This does not mean that all transcriptions are fully done.
    
    The input chunks may optionally contain audio timings (for example, StreamingAudioSender adds this
    information), but they are generally not used. Also, some timestamps are automatically filled:
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
        This method will be called in a separate thread on `self.start_thread()` and should live forever
        (use `self.input_buffer.get()` in the `while True` loop). To get the next input chunk, we can use
        `self.input_buffer.get()` (blocks until the next chunk is available). To emit a new output chunk,
        we can use `self.output_buffer.put()` (non-blocking).
        
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


LATEST = '__latest__'  # a special symbol to refer to the latest transcription chunk


@dataclass(kw_only=True)
class TranscriptionChunk:
    """
    A chunk returned by a streaming ASR model, may contain any text and any ID. If the model
    wants to edit the previous chunk, it can yield the same ID with another text, or refer to
    the last chunk with ID == LATEST. Example:
    
    TranscriptionChunk.join([
        TranscriptionChunk(text='a'),               # append a new chunk with text 'a' without an explicit id to refer
        TranscriptionChunk(id=LATEST, text='a2'),   # edit the latest chunk: 'a' -> 'a2'
        TranscriptionChunk(id=1, text='b'),         # append a new chunk with text 'b', 2 chunks in total: 'a', 'b'[id=1]
        TranscriptionChunk(id=2, text='c'),         # append a new chunk with text 'c', 3 chunks in total: 'a', 'b'[id=1], 'c'[id=2]
        TranscriptionChunk(id=1, text='b2 b3'),     # edit the chunk with id=1: 'a', 'b2 b3'[id=1], 'c'[id=2]
    ]) == 'a2 b2 b3 c'
    """
    ref: int | str = field(default_factory=lambda: str(uuid.uuid4()))
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
            if t.ref == LATEST:
                # edit the lastest chunk
                current_id = list(parts)[-1] if len(parts) else '<initial>'
            else:
                # edit one of the previous chunks or add a new chunk, set as latest
                current_id = t.ref
            
            parts[current_id] = t.text
            
        return ' '.join(parts.values())