"""Streaming ASR pipelines construction and evaluation.

Model
==========

A streaming pipeline starts with subclassing a
:class:`~asr_eval.streaming.model.StreamingASR` model. It runs in a
separate thread, accepts a stream of
:class:`~asr_eval.streaming.model.InputChunk` and emits a stream of
:class:`~asr_eval.streaming.model.OutputChunk`. Each chunk contains
audio recording id, which allows to process multiple recordings in
parallel.

**Input format**. A model may accept different audio formats (float32,
WAV etc.) and sample rates - this is defined in the
:attr:`~asr_eval.streaming.model.StreamingASR.audio_type` and
:attr:`~asr_eval.streaming.model.StreamingASR.sampling_rate` fields.

**Output format**. A each output chunk contains a
:class:`~asr_eval.streaming.model.TranscriptionChunk` which carries a
text and a chunk ID. Re-emitting new chunk with the same ID allows to
correct a partial transcription emitted earlier.

Sending and receiving
=========================

One can directly send and receive chunks from
:class:`~asr_eval.streaming.model.StreamingASR`. These methods can be
connected to web API routes in production.

To evaluate, several automation tools exist:

**Sending**. A :func:`~asr_eval.streaming.evaluation.make_sender`
tool automates sending. It creates a list of
:class:`~asr_eval.streaming.sender.Cutoff` that define a schedule for
sending audio chunks. Then a
:class:`~asr_eval.streaming.sender.StreamingSender` is instantiated.
You can further call :code:`sender.start_sending()` to start sending.

**Receiving**. A receiving outputs can be automated via
:func:`~asr_eval.streaming.evaluation.receive_transcription`.

Remapping
=========================

Remapping is an optional mechanism that eliminates time spans where both
the sender waits (due to its schedule) and the model waits (because it
already processed the chunk and waits for the next). This makes
evaluation faster than real time. A remapping
:func:`~asr_eval.streaming.evaluation.remap_time` is the operation
that correct the timings as if the sending were in real time. However,
that this is not applicable (would work incorrectly) for
:code:`StreamingASR` that start another threads from its main thread
(where :attr:`~asr_eval.streaming.model.StreamingASR.is_multithreaded`
is True).

Evaluating
=========================

To evaluate, one needs audio samples with word timings. This can be done
via forced alignment using a
:class:`~asr_eval.models.base.interfaces.CTC` model and
:func:`~asr_eval.align.timings.fill_word_timings_inplace` function.

The :func:`~asr_eval.streaming.evaluation.evaluate_streaming` function
automates evaluation. It analyses a history of input and output chunks
(this history can also be serialized to json and loaded back). As a
result, we obtain a
:class:`~asr_eval.streaming.evaluation.RecordingStreamingEvaluation`
object.

Further we can draw various diagrams, such as:

- :func:`~asr_eval.streaming.plots.draw_partial_alignment`
- :func:`~asr_eval.streaming.plots.partial_alignments_plot`
- :func:`~asr_eval.streaming.plots.visualize_history`
- :func:`~asr_eval.streaming.plots.streaming_error_vs_latency_histogram`
- :func:`~asr_eval.streaming.plots.latency_plot`
- :func:`~asr_eval.streaming.plots.show_last_alignments`

This package also features two wrappers that convert a non-streaming
model into a streaming one
(:class:`~asr_eval.streaming.wrappers.OfflineToStreaming`) or back
(:class:`~asr_eval.streaming.wrappers.StreamingToOffline`).
"""