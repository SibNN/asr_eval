Code tree
###################

The asr_eval corrently consists of 7000 lines of code and provides about 140 functions
and classes to perform ASR prediction and evaluation.

File tree:

.. code-block:: bash

    asr_eval
    ├── align
    │   ├── data.py
    │   ├── dynprog.py
    │   ├── multiple.py
    │   ├── parsing.py
    │   ├── partial.py
    │   ├── plots.py
    │   ├── recursive.py
    │   └── timings.py
    ├── bench
    │   ├── check.py
    │   ├── dashboard.py
    │   ├── datasets.py
    │   ├── evaluator.py
    │   ├── pipelines.py
    │   ├── README.md
    │   ├── recording.py
    │   └── run.py
    ├── correction
    │   ├── bow_corpus.py
    │   ├── corrector_langchain.py
    │   ├── corrector_levenshtein.py
    │   ├── corrector_wikirag.py
    │   └── interfaces.py
    ├── ctc
    │   ├── base.py
    │   └── forced_alignment.py
    ├── linguistics
    │   └── linguistics.py
    ├── models
    │   ├── ast_wrapper.py
    │   ├── base
    │   │   ├── interfaces.py
    │   │   ├── longform.py
    │   │   └── openai_wrapper.py
    │   ├── flamingo_wrapper.py
    │   ├── gemma_wrapper.py
    │   ├── gigaam_wrapper.py
    │   ├── legacy_pisets_wrapper.py
    │   ├── nvidia_conformer_wrapper.py
    │   ├── pisets_wrapper.py
    │   ├── pyannote_segmenter.py
    │   ├── pyannote_wrapper.py
    │   ├── qwen2_audio_wrapper.py
    │   ├── qwen_audio_wrapper.py
    │   ├── speechbrain_wrapper.py
    │   ├── t_one_wrapper.py
    │   ├── vosk54_wrapper.py
    │   ├── vosk_streaming_wrapper.py
    │   ├── voxtral_wrapper.py
    │   ├── whisper_wrapper.py
    │   └── yandex_speechkit_wrapper.py
    ├── py.typed
    ├── segments
    │   ├── chunking.py
    │   └── segment.py
    ├── streaming
    │   ├── buffer.py
    │   ├── caller.py
    │   ├── evaluation.py
    │   ├── model.py
    │   ├── plots.py
    │   └── sender.py
    ├── tts
    │   └── yandex_speechkit.py
    └── utils
        ├── audio_ops.py
        ├── autodoc.py
        ├── formatting.py
        ├── misc.py
        ├── plots.py
        ├── serializing.py
        ├── server.py
        ├── srt.py
        └── types.py
    tests
    ├── test_align.py
    ├── test_buffer.py
    ├── test_ctc.py
    ├── testdata
    │   ├── formula1.mp3
    │   ├── formula1.txt
    │   ├── long.mp3
    │   ├── minecraft.txt
    │   ├── minecraft.wav
    │   ├── podlodka_test_0.wav
    │   └── vosk.wav
    ├── test_evaluation.py
    ├── test_gigaam.py
    ├── test_model.py
    ├── test_speechbrain.py
    ├── test_t_one.py
    ├── test_transcription.py
    ├── test_vosk.py
    └── test_wave.py