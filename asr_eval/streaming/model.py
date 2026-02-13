from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Literal, Self, Sequence, TypeVar, override

import numpy as np

from asr_eval.streaming.buffer import ID_TYPE, StreamingQueue
from asr_eval.utils.misc import new_uid
from asr_eval.utils.types import FLOATS, INTS


__all__ = [
    'Signal',
    'Exit',
    'AUDIO_CHUNK_TYPE',
    'CHUNK_TYPE',
    'InputChunk',
    'OutputChunk',
    'check_consistency',
    'ASRStreamingQueue',
    'InputBuffer',
    'OutputBuffer',
    'StreamingASR',
    'DummyASR',
    'TranscriptionChunk',
]


class Signal(Enum):
    """Signals to control StreamingASR thread. See
    :class:`~asr_eval.streaming.model.StreamingASR` docs for details.
    """

    FINISH = 0
    """Indicates that stream finishes for the given audio recording."""


class Exit(Exception):
    """A signal to terminate StreamingASR thread. See
    :class:`~asr_eval.streaming.model.StreamingASR` docs for details.
    """

    pass


AUDIO_CHUNK_TYPE = FLOATS | INTS | bytes
"""Audio stream that is chunkable using slices."""
    

@dataclass(kw_only=True)
class InputChunk:
    """ An input chunk for
    :class:`~asr_eval.streaming.model.StreamingASR`. Input chunks can be
    sent by :class:`~asr_eval.streaming.sender.StreamingSender` or
    manually and received by :code:`StreamingASR` thread.
    
    See :class:`~asr_eval.streaming.model.StreamingASR` docs for usage
    details.
    """

    data: AUDIO_CHUNK_TYPE | Literal[Signal.FINISH]
    """Either a chunk of audio stream, or a :code:`Signal.FINISH`."""
    
    end_time: float
    """A chunk end time (in seconds) in the audio timescale, where 0
    means the beginning of the audio recording.
    """
    
    put_timestamp: float = np.nan
    """Is filled automatically when the chunk is added to the
    input buffer.
    """

    get_timestamp: float = np.nan
    """Is filled automatically when the :code:`StreamingASR` thread
    takes the chunk from the buffer.
    """
 
    
@dataclass(kw_only=True)
class OutputChunk:
    """ An output chunk for
    :class:`~asr_eval.streaming.model.StreamingASR`. Output chunks are
    sent by :code:`StreamingASR` thread and received manually or
    by :func:`~asr_eval.streaming.evaluation.receive_transcription`.
    
    See :class:`~asr_eval.streaming.model.StreamingASR` and
    :func:`~asr_eval.streaming.evaluation.receive_transcription` docs
    for usage details.
    """

    data: TranscriptionChunk | Literal[Signal.FINISH]
    """Either a part of transcription, or a :code:`Signal.FINISH`."""
    
    seconds_processed: float
    """A total audio seconds processed before emitting the current
    chunk. Is filled by the transcriber.
    """
    
    put_timestamp: float = np.nan
    """Is filled automatically when the chunk is added to the output
    buffer.
    """

    get_timestamp: float = np.nan
    """Is filled automatically when the chunk is taken from the output
    buffer.
    """
    
    
def check_consistency(
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk],
):
    """ Asserts that:

    1. For each input and output chunk
       :code:`put_timestamp <= get_timestamp`.
    2. For each output chunk :code:`seconds_processed` is not larger
    than audio seconds taken from the input buffer by the time the
    output is put into the buffer.
    
    Fails indicate errors in the chunk processing pipeline (sender,
    buffer or model).

    Raises:
        AssertionError: On check failures.
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
"""Either input or output chunk."""


class ASRStreamingQueue(StreamingQueue[CHUNK_TYPE]):
    """An input or output buffer in
    :class:`~asr_eval.streaming.model.StreamingASR`.
    
    This subclass extends the
    :class:`~asr_eval.streaming.buffer.StreamingQueue`:
     
    1. It fills :code:`put_timestamp`, :code:`get_timestamp` for input
    or output chunks.
    2. It asserts that if :code:`Signal.FINISH` was received, no more
    chunks are expected for this audio recording ID.
    """
    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._counts: dict[ID_TYPE, int] = defaultdict(int)
        self._finished_ids: set[ID_TYPE] = set()
    
    @override
    def get(
        self, id: ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[CHUNK_TYPE, ID_TYPE]:
        data, id = super().get(id=id, timeout=timeout)
        data.get_timestamp = time.time()
        return data, id
    
    @override
    def put(self, data: CHUNK_TYPE, id: ID_TYPE = 0) -> None:
        with self._condition:
            self._validate(data=data, id=id)
            data.put_timestamp = time.time()
            self._counts[id] += 1
            return super().put(data, id=id)
    
    def _validate(self, data: CHUNK_TYPE, id: ID_TYPE):
        try:
            assert id not in self._finished_ids, (
                f'Got id={id} but already finished this id'
            )
            if data.data is Signal.FINISH:
                self._finished_ids.add(id)
        except BaseException as e:
            # this will raise the error in the receiver thread on .get()
            self.put_error(e)
            # raise the error in the sender thread
            raise e


class InputBuffer(ASRStreamingQueue[InputChunk]):
    """ An input buffer for
    :class:`~asr_eval.streaming.model.StreamingASR`.

    This subclass adds a
    :meth:`~asr_eval.streaming.model.InputBuffer.get_with_rechunking`
    method. If it was called at least once, a rechunking mode is enabled
    and :code:`.get()` cannot be called anymore.
    """

    def __init__(self, name: str = 'unnamed'):
        super().__init__(name=name)
        self._rechunking_mode_on: bool = False
        self._audio_buffer: dict[ID_TYPE, AUDIO_CHUNK_TYPE] = {}
        self._rechunking_end_time: dict[ID_TYPE, float] = {}
        self._rechunking_finish_received: dict[ID_TYPE, bool] = {}
        
    @override
    def get(
        self, id: ID_TYPE | None = None, timeout: float | None = None
    ) -> tuple[InputChunk, ID_TYPE]:
        assert not self._rechunking_mode_on, (
            'Method .get_with_rechunking() was called'
            ' earlier. After this, you cannot call .get()'
            ' anymore because some data could have been'
            ' consumed by the rechunking buffer.'
        )
        return super().get(id=id, timeout=timeout)
    
    # def wait_until_available(
    #     self,
    #     size: int,
    #     id: ID_TYPE | None = None,
    # ) -> tuple[ID_TYPE, int, bool, bool]:
    #     '''
    #     If ID is specified, waits until the given size is available or the Signal.FINISH is received for the
    #     given ID. If ID is not specified, waits until the given size is available for any ID, or the
    #     Signal.FINISH is received for any ID.
        
    #     Returns:
    #     - The ID (the same ID as the `id` argument if specified),
    #     - The available size - it may be larger than requested, if more data is available in buffer, or
    #       may be smaller if the Signal.FINISH is received for this ID.
    #     - The flag if Signal.FINISH is received
    #     - The flag if the available size is less than the requested (may be true only if Signal.FINISH
    #       is received).
    #     '''
    
    def get_with_rechunking(
        self,
        size: int,
        id: ID_TYPE | None = None,
    ) -> tuple[ID_TYPE, AUDIO_CHUNK_TYPE | None, bool, float]:
        """ Internally calles ;code:`.get()` as many times as needed and
        concatenates and/or slices the results to obtain the desired
        array size.
        
        For example, let each input chunk contain 1000 audio frames, and
        we requested :code:`size=2400`. The :code:`.get()` will be
        called 3 times, and the last chunks will be split into two
        parts, of size 400 and 600. An array of size 2400 will be
        returned, and 600 remaining elements will be kept in the
        rechunking buffer. If then we :code:`request size=100`, the
        array of size 100 will be returned without new :code:`get()`,
        and buffer will keep 500 remaining elements, and so on.
        
        The retuned array can be smaller than requested only if
        :code:`Signal.FINISH` reached for the ID.
        
        Returns:
            1. ID (equals the :code:`id` argument if was specified,
               otherwise the first available id).
            2. Audio chunk of the desired size (or less if
               :code:`Signal.FINISH` reached).
            3. A flag if :code:`Signal.FINISH` reached.
            4. The audio end time of the last recived chunk (even if its
               part is still in the rechunking buffer). TODO maybe set
               the audio end time more correctly?
        """

        self._rechunking_mode_on = True
        while True:
            # do we have data to return?
            if id is not None:
                if (
                    (id in self._audio_buffer \
                        and len(self._audio_buffer[id]) >= size)
                    or self._rechunking_finish_received.get(id, False)
                ):
                    break
            else:
                # loop over buffer and find id to return
                for _id, is_finished in (
                    self._rechunking_finish_received.items()
                ):
                    if is_finished:
                        id = _id
                        break
                for _id, buffered_chunk in self._audio_buffer.items():
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
                if recieved_id in self._audio_buffer:
                    if isinstance(chunk.data, (bytes, str)):
                        self._audio_buffer[recieved_id] += chunk.data # type: ignore
                    else:
                        # numpy array
                        self._audio_buffer[recieved_id] = np.concatenate([
                            self._audio_buffer[recieved_id], chunk.data
                        ])
                else:
                    self._audio_buffer[recieved_id] = chunk.data
              
        # we will return data from buffer for the given id
        assert id is not None
                
        if not id in self._audio_buffer:
            # no data in buffer, is_finished == True
            assert self._rechunking_finish_received[id]
            chunk_to_return = None
            is_finished = True
        elif len(self._audio_buffer[id]) > size:
            # more than enough data in buffer, is_finished or not
            chunk_to_return = self._audio_buffer[id][:size]
            self._audio_buffer[id] = self._audio_buffer[id][size:]
            is_finished = False
        else:
            # just enough or less data in buffer, is_finished or not
            chunk_to_return = self._audio_buffer[id]
            is_finished = self._rechunking_finish_received.get(id, False)
            del self._audio_buffer[id]
            
        end_time = self._rechunking_end_time.get(id, 0)
        if is_finished:
            del self._rechunking_finish_received[id]
            self._rechunking_end_time.pop(id, None)
        
        return id, chunk_to_return, is_finished, end_time

class OutputBuffer(ASRStreamingQueue[OutputChunk]):
    """An output buffer for StreamingASR."""


class StreamingASR(ABC):
    """An abstract streaming transcriber that is able to process
    multiple audio recordings in parallel.
    
    Accepts a stream of input chunks marked by recording ID and emits a
    stream of output chunks.
    
    **Definitions:**
    
    - **Audio chunk**: a part of a waveform. Sampling rate is defined in
      the class constructor. For example, a 10 sec mono recording with
      rate 16_000 can be represented as 10 chunks, each with shape
      :code:`(16_000,)`. Several channels can also be supported for some
      models. The chunk length is not restricted. NOTE: each
      :code:`StreamingASR` implementation has
      :attr:`~asr_eval.streaming.model.StreamingASR.audio_type` and
      :attr:`~asr_eval.streaming.model.StreamingASR.sampling_rate`
      fields that define a required type and sampling rate of audio
      chunks.
    - **TranscriptionChunk**: a partial transcription that either add
      new words to the transcription or edit the previous words. See
      details in the
      :class:`~asr_eval.streaming.model.TranscriptionChunk` docs.
    - **Recording ID**: a unique int or string identifier for an audio
      recording. This is useful if several recordings are streamed
      simultaneously, and we should know which audio recording each
      chunk belongs to. IDs should be unique for :code:`StreamingASR`
      object and should not be reused, or exception will be thrown.
    - **Signal.FINISH**: a symbol that signals that a stream for a
      specific recording ID has ended. This can refer to either the
      input stream (audio chunks) or the output stream
      (:code:`TranscriptionChunk`s).
    - **Exit**: an exception that signals that all streams for all
      recording IDs have ended. This can refer to either the input
      stream or the output stream. After receiving
      :exc:`~asr_eval.streaming.model.Exit` from the input buffer and
      sending :code:`Exit` to the output buffer, :code:`StreamingASR`
      thread finishes.
    
    **Data model:**
    
    Each input chunk can be one of:

    1. An :code:`InputChunk(id=<Recording ID>, data=<Audio chunk>)`.
    2. An :code:`InputChunk(id=<Recording ID>, data=Signal.FINISH)` -
       indicates that the audio for the ID has been fully sent.
    
    Each output chunk can be one of:

    1. An :code:`OutputChunk(id=<Recording ID>, data=<TranscriptionChunk>)`.
    2. An :code:`OutputChunk(id=<Recording ID>, data=Signal.FINISH)` -
       indicates that FINISH input chunk received fhr the given ID and
       the transcription is done.
    
    Models may fill :code:`.seconds_processed` field in
    :class:`~asr_eval.streming.model.OutputChunk` - audio seconds
    processed (for the current recording ID) before yielding the current
    output chunk. This may be useful, because we could send 100 chunks
    (let it be 10 sec in total), but the model performs slow
    calculations and has already processed only 20 chunks (2 sec in
    total). Depending on the testing scenario we can treat the result as
    a partial transcription of the first 2 or 10 seconds of the audio
    signal.
    
    **Sending and receiving:**
    
    After creating an :code:`StreamingASR` object, we should start a
    thread that will process input chunks and emit output chunks. After
    this, new audio chunks can be sent using
    :code:`.input_buffer.put(...)` (non-blocking), and the outputs can
    be received with :code:`.output_buffer.get(...)` (blocks until
    output becomes available). Instead of manual sending, a
    :func:`~asr_eval.streaming.evaluation.make_sender` function
    can be helpful. It prepares a sender to send audio chunks with a
    delay between each chunk.
    
    Input and output buffers automatically fill the follwing fields:

    1. :code:`InputChunk.put_timestamp` - the time when the chunk added
       to the :code:`StreamingASR.input_buffer`.
    2. :code:`InputChunk.get_timestamp` - the time when the chunk
       received from the :code:`StreamingASR.input_buffer`.
    3. :code:`OutputChunk.put_timestamp` - the time when the chunk added
       to the :code:`StreamingASR.output_buffer`.
    4. :code:`OutputChunk.get_timestamp` - the time when the chunk
       received from the :code:`StreamingASR.output_buffer`.
    
    Pts 1, 4 happen in the caller thread, and pts 2, 3 happen in the
    :code:`StreamingASR` thread.
    
    **Terminating a StreamingASR thread:**
    
    An :exc:`~asr_eval.streaming.model.Exit` exception in the input
    buffer indicates that all audios have been fully sent. An
    :exc:`~asr_eval.streaming.model.Exit` exception in the output buffer
    indicates that :code:`Exit` was received from the input buffer and
    the :code:`StreamingASR` thread exited. This does not mean that all
    transcriptions are fully done.
    
    **Exception handling:**
    
    1. Any exception raised from the :code:`StreamingASR`:code:` thread
       will set the output buffer in the error state. This will raise
       the exception when reading from the output buffer.
    2. Trying to write invalid data into the input buffer (including
       reusing previous IDs) may set it into the error state. This will
       raise the exception when reading from the input buffer in the
       :code:`StreamingASR`. thread, then see pt. 1.
    3. Exceptions in the sender thread will set the input buffer into
       the error state, then see pt. 2.
    4. :exc:`~asr_eval.streaming.model.Exit` is a special exception type
       indicating that input or output stream has been closed properly.
    
    **Implementing models:**
    
    To subclass a :code:`StreamingASR`, one should implement
    :meth:`~asr_eval.streaming.model.StreamingASR._run` and
    :attr:`~asr_eval.streaming.model.StreamingASR.audio_type` methods.
    Also, a subclass :code:`__init__` method shoud call
    :code:`super().__init__` specifying the audio sampling rate.
    """

    sampling_rate: int
    """Sampling rate for the input audio chunks.
    
    TODO clarify what to set for bytes or WAV.
    """
    
    def __init__(self, sampling_rate: int = 16_000):
        self.sampling_rate = sampling_rate
        self._thread: threading.Thread | None = None
    
    def start_thread(self) -> Self:
        """Start the background thread with
        :meth:`~asr_eval.streaming.model.StreamingASR._run` that
        processes input chunks and emits outputs chunks.
        """
        
        assert self._thread is None
        self.input_buffer = InputBuffer(name='input buffer')
        self.output_buffer = OutputBuffer(name='output buffer')
        self._thread = threading.Thread(
            target=self._run_and_send_exit, daemon=True
        )
        self._thread.start()
        return self
    
    def stop_thread(self) -> None:
        """Stops the background thrad started with
        :meth:`~asr_eval.streaming.model.StreamingASR.start_thread`.
        """

        assert self._thread
        self.input_buffer.put_error(Exit())
        self._thread.join()
        self._thread = None
    
    def is_thread_started(self) -> bool:
        """Is the background thread started with
        :meth:`~asr_eval.streaming.model.StreamingASR.start_thread`
        running?
        """

        return  self._thread is not None
    
    def _run_and_send_exit(self):
        try:
            assert self._thread and self._thread.ident == threading.get_ident()
            self._run()
            # if we are here, _run has ended without Exit exception,
            # this is not usual
            self.output_buffer.put_error(Exit())
        except BaseException as e:
            if isinstance(e, RuntimeError) and isinstance(e.__cause__, Exit):
                self.output_buffer.put_error(Exit())
                # if we are here, _run has ended after receiving
                # Exit from the input stream, this is usual
            else:
                # if we are here, _run has ended with a different
                # exception, maybe originating from the
                # input buffer or the problem in the _run method
                # itself. We propagate the exception to the
                # output buffer and re-raise
                self.output_buffer.put_error(e)
                raise e
    
    @abstractmethod
    def _run(self):
        """A background thread that processes input chunks and emits
        outputs chunks.

        Is started with
        :meth:`~asr_eval.streaming.model.StreamingASR.start_thread` and
        should live forever, usually with :code:`while True` loop.
        To get the next input chunk, we can use
        :code:`self.input_buffer.get()` or
        :meth:`~asr_eval.streaming.model.InputBuffer.get_with_rechunking`
        (both methods block until the next chunk is available). To emit
        a new output chunk, use :code:`self.output_buffer.put()`
        (non-blocking).
        
        For example, if 16_000 floats/sec are streamed, and an exteral
        sender sends chunks of size 1600 10 times per second, but your
        model want to get 1s chunks, call
        `self.input_buffer.get_with_rechunking(size=16_000)`. This will
        block until 10 chunks are accumulated for any ID and return the
        result.
        
        Normally on
        :meth:`~asr_eval.streaming.model.StreamingASR.stop_thread` an
        :exc:`~asr_eval.streaming.model.Exit` exception is raised when
        the :code:`_run` method tries to read from the input buffer. It
        causes exit from :code:`_run` and is handled in a wrapping
        method :code:`_run_and_send_exit`. So, the :code:`Exit`
        exception should not be handled in :code:`_run`.

        :meta public:
        """
    
    @property
    @abstractmethod
    def audio_type(self) -> Literal['float', 'int', 'bytes', 'wav']:
        """The required input audio format. Together with
        :attr:`~asr_eval.streaming.model.InputBuffer.sampling_rate`
        property, forms a specification of input audio.
        
        See also :func:`~asr_eval.utils.audio_ops.convert_audio_format`
        for details about formats.
        """
    
    @property
    def is_multithreaded(self) -> bool:  # for remap_time, TODO docs
        """Whether another threads are started from the background
        thread :meth:`~asr_eval.streaming.model.StreamingASR._run`.
        
        False by defatult. If overriden with True, the evaluation
        protocol will not try to use
        :func:`~asr_eval.streaming.evaluation.remap_time`."""

        return False



class DummyASR(StreamingASR):
    """Will transcribe N seconds long audio into "1 2 ... N"."""
    
    def __init__(self, sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._received_seconds: dict[ID_TYPE, float] = defaultdict(float)
        self._transcribed_seconds: dict[ID_TYPE, int] = defaultdict(int)
    
    @override
    def _run(self):
        while True:
            chunk, id = self.input_buffer.get()
            
            self._received_seconds[id] += (
                len(chunk.data) / self.sampling_rate
                if chunk.data is not Signal.FINISH
                else 0
            )
            
            new_transcribed_seconds = math.ceil(self._received_seconds[id])
            
            for i in range(
                self._transcribed_seconds[id], new_transcribed_seconds
            ):
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
    """ A chunk returned by a
    :class:`~asr_eval.streaming.model.StreamingASR` model.
    
    Contain a text and optional ID. If we want to override the
    previously emitted partial transcription, we should emit a new chunk
    with the same ID. It will not be treated as a new text to append,
    but as a correction for the previous chunk with this ID.

    Note:
        A :code:`TranscriptionChunk` id is not the same concept as audio
        recording ID.
    
    Example:
        >>> from asr_eval.streaming.model import TranscriptionChunk
        >>> chunks = []
        >>> # append a new chunk without an explicit uid to refer
        >>> # without ID we cannot correct this chunk later
        >>> chunks.append(TranscriptionChunk(text='word1'))
        >>> # append a new chunk with id 1
        >>> chunks.append(TranscriptionChunk(uid=1, text='word2'))
        >>> # append a new chunk with id 2
        >>> chunks.append(TranscriptionChunk(uid=2, text='word3'))
        >>> print(TranscriptionChunk.join(chunks))
        word1 word2 word3
        >>> # correct a chunk with id 1
        >>> chunks.append(TranscriptionChunk(uid=1, text='word2a word2b'))
        >>> # remove a chunk with id 2
        >>> chunks.append(TranscriptionChunk(uid=2, text=''))
        >>> print(TranscriptionChunk.join(chunks))
        word1 word2a word2b
    """
    uid: int | str = field(default_factory=new_uid)
    text: str
    
    @classmethod
    def join(
        cls,
        transcriptions: (
            Sequence[TranscriptionChunk]
            | Sequence[OutputChunk | Literal[Signal.FINISH]]
        ),
    ) -> str:
        """Join transcription chunks. If the :code:`transcriptions`
        are :code:`OutputChunk` instances, extracts a transcription
        chunks from each output chunk.
        
        See example in the
        :class:`~asr_eval.streaming.model.TranscriptionChunk` docs.
        """
        parts: dict[int | str, str] = {}
        
        for t in transcriptions:
            if isinstance(t, OutputChunk):
                t = t.data
            if t is Signal.FINISH:
                continue
            parts[t.uid] = t.text
            
        return ' '.join([v for v in parts.values() if len(v)])