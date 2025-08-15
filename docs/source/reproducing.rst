Reproducing experiments
###################

.. code-block:: bash
    
    DATASETS="-d golos-farfield podlodka-full resd fleurs speech-massive common-voice-17.0 -m 200"

    tmp/venv_whisper/bin/python -m asr_eval.bench.run -p whisper-large-v3 $DATASETS
    tmp/venv_gigaam/bin/python -m asr_eval.bench.run -p gigaam-ctc $DATASETS
    tmp/venv_gigaam/bin/python -m asr_eval.bench.run -p gigaam-rnnt-vad $DATASETS
    tmp/venv_flamingo/bin/python -m asr_eval.bench.run -p flamingo-ru-vad $DATASETS
    tmp/venv_tone/bin/python -m asr_eval.bench.run -p t-one-vad $DATASETS
    tmp/venv_vosk/bin/python -m asr_eval.bench.run -p vosk-0.54-vad $DATASETS
    tmp/venv_yandex_speechkit/bin/python -m asr_eval.bench.run -p yandex-speechkit $DATASETS
    # tmp/venv_voxtral/bin/python -m asr_eval.bench.run -p voxtral-3B $DATASETS
    tmp/venv_qwen2audio/bin/python -m asr_eval.bench.run -p qwen2-audio-vad $DATASETS

    python -m asr_eval.bench.dashboard