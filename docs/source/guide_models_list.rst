Models list
###############################

*asr_eval* provides more than 50 various built-in model configurations
for speech recognition, and you can add custom ones.

For now, in early development, we focused mainly on Russian models,
but there are also multilingual ones.

.. |br| raw:: html

   <br>

.. admonition:: Adding models

    You can add new models
    locally, as described in :ref:`Framework quickstart <framework_quickstart>`.
    You also can suggest new models by opening an issue or pull request.

    To be able to wrap or combine components, we suggest implementing:
    
    - :class:`~asr_eval.models.base.interfaces.CTC` interface for CTC models.
    - :class:`~asr_eval.models.base.interfaces.TimedTranscriber` interface for ASR models that return text with timings.
      |br| Return :class:`~asr_eval.segments.TimedDiarizationText` for models with diarization.
    - :class:`~asr_eval.models.base.interfaces.Transcriber` interface for
      ASR models that return text without timings.
    - :class:`~asr_eval.streaming.model.StreamingASR` interface for
      streaming ASR modeels.
    - :class:`~asr_eval.models.base.interfaces.Segmenter` interface for
      voice activity detection models.

Whisper pipelines
==========================

Base and fine-tuned checkpoints, Faster-Whisper.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/whisper.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.whisper._

Wav2vec2 pipelines
==========================

Various checkpoints from Hugging Face.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/wav2vec2.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.wav2vec2._

NVIDIA NeMo pipelines
==========================

Canary, Parakeet, Conformer, FastConformer.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/nemo.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.nemo._

Audio-LLM pipelines
=====================

Voxtral, Vikhr Borealis, Flamingo, Gemma3n, Qwen2-audio.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/audio_llm.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.audio_llm._

SpeechBrain pipelines
==========================

Streaming GigaSpeech conformer.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/speechbrain.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.speechbrain._

.. _gigaam:

GigaAM pipelines
=====================

GigaAM v2, v3 versions, CTC and RNNT, with with end-to-end punctuation.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/gigaam.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.gigaam._

Vosk pipelines
==========================

Vosk 0.42 streaming and 0.54 non-streaming.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/vosk.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.vosk._

T-One pipelines
==========================

T-One steaming model.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/t_one.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.t_one._

API pipelines
==================

Yandex SpeechKit, Salute API.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/api.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.api._

Pisets pipelines
==========================

A wav2vec + Whisper pipeline.

📄 See `source code <_modules/asr_eval/bench/pipelines/_registered/pisets.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.pisets._

Composite and experimental pipelines
======================================

See `source code <_modules/asr_eval/bench/pipelines/_registered/composite.html>`_ for definitions.

.. only:: hidden

    .. autoclass:: asr_eval.bench.pipelines._registered.composite._
