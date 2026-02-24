# Class StreamingASR (defined in asr_eval/streaming/model.py at lines 354-594)

class StreamingASR(abc.ABC):
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
    ...

    sampling_rate: int
    """Sampling rate for the input audio chunks.

    TODO clarify what to set for bytes or WAV.
    """

    def start_thread(self) -> typing.Self:
        """Start the background thread with
        :meth:`~asr_eval.streaming.model.StreamingASR._run` that
        processes input chunks and emits outputs chunks.
        """
        ...

    def stop_thread(self) -> None:
        """Stops the background thrad started with
        :meth:`~asr_eval.streaming.model.StreamingASR.start_thread`.
        """
        ...

    def is_thread_started(self) -> bool:
        """Is the background thread started with
        :meth:`~asr_eval.streaming.model.StreamingASR.start_thread`
        running?
        """
        ...

    @property
    @abc.abstractmethod
    def audio_type(self) -> typing.Literal['float', 'int', 'bytes', 'wav']:
        """The required input audio format. Together with
        :attr:`~asr_eval.streaming.model.InputBuffer.sampling_rate`
        property, forms a specification of input audio.

        See also :func:`~asr_eval.utils.audio_ops.convert_audio_format`
        for details about formats.
        """
        ...

    @property
    def is_multithreaded(self) -> bool:  # for remap_time, TODO docs
        """Whether another threads are started from the background
        thread :meth:`~asr_eval.streaming.model.StreamingASR._run`.

        False by defatult. If overriden with True, the evaluation
        protocol will not try to use
        :func:`~asr_eval.streaming.evaluation.remap_time`."""
        ...