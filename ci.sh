# test minimal config
uv sync
uv pip install pytest
uv run python -m pytest tests/test_imports.py

# tests
uv sync --extra all --extra dev
uv run python -m pytest tests

# doctests
uv sync --extra all --extra dev
uv pip install git+https://github.com/salute-developers/GigaAM
uv run python -m xdoctest asr_eval --verbose 1
uv run sphinx-build -M doctest docs/source docs/build docs/source/guide_alignment_wer.rst
uv run sphinx-build -M doctest docs/source docs/build docs/source/guide_streaming_evaluation.rst
uv run sphinx-build -M doctest docs/source docs/build docs/source/guide_evaluation_dashboard.rst

# type checking
uv sync --extra all --extra dev
uv pip install -r installation/type_checking.txt --no-deps
uv run python -m pyright -p pyrightconfig.json asr_eval

# docs building
uv sync --extra dev --extra dev
rm -rf docs/build
uv run sphinx-build docs/source docs/build
(cd docs && uv run make simplepdf)

# whisper, wav2vec, vikhr
uv sync --extra models_stable --extra datasets
uv run python -m asr_eval.bench.check whisper-tiny --audio EN_LONG
uv run python -m asr_eval.bench.check wav2vec2-large-960h --audio EN_LONG
uv run python -m asr_eval.bench.check vikhr-borealis-vad --audio RU_LONG

# wav2vec + LM
uv sync --extra models_stable --extra datasets --extra kenlm
uv pip install pyctcdecode==0.5.0 --no-deps
uv run python -m asr_eval.bench.check wav2vec2-large-ru-golos-lm-t-one --audio RU_LONG

# gigaam + LM (ru only)
uv sync --extra models_stable --extra datasets --extra kenlm
uv pip install git+https://github.com/salute-developers/GigaAM
uv pip install pyctcdecode==0.5.0 --no-deps
uv run python -m asr_eval.bench.check gigaam3-rnnt-vad --audio RU_LONG
uv run python -m asr_eval.bench.check gigaam2-ctc-lm-t-one --audio RU_LONG

# vosk (ru only)
sudo apt install cmake -y
uv sync --extra datasets
uv pip install -r installation/vosk.txt
uv run python -m asr_eval.bench.check vosk-0.54-vad --audio RU_LONG
uv run python -m asr_eval.bench.check vosk-ru-0.42-streaming --audio RU

# t-one (ru only)
# T-One depends on pyctcdecode, and pyctcdecode (PyPI version) depends on numpy 1.x
# this is too restrictive, given that pyctcdecode also works perfectly for numpy 2.x
# to address this, we first install T-One and pyctcdecode without dependencies,
# then install dependencies manually
uv sync --extra models_stable --extra datasets --extra kenlm
uv pip install pyctcdecode==0.5.0 --no-deps
uv pip install git+https://github.com/voicekit-team/T-one --no-deps
uv pip install "huggingface-hub (>=0.14,<1)" "onnxruntime (>=1.12,<2)"
uv run python -m asr_eval.bench.check t-one-vad --audio RU_LONG

# voxtral
uv sync --extra models_stable --extra datasets --extra openai
uv run python -m asr_eval.bench.check voxtral-3B

# nemo
uv sync --extra models_stable --extra datasets --extra nemo
uv run python -m asr_eval.bench.check parakeet-tdt-0.6b-v3-vad --audio EN_LONG

# speechbrain
uv sync --extra models_stable --extra datasets
uv pip install speechbrain "huggingface_hub<1"
uv run python -m asr_eval.bench.check speechbrain-conformer-gigaspeech-streaming

# gemma3n
uv sync --extra models_stable --extra datasets
uv pip install -r installation/gemma3n.txt
uv run python -m asr_eval.bench.check gemma3n-vad

# qwen2audio
uv sync --extra models_stable --extra datasets --extra qwen2audio
uv pip install flash-attn --no-build-isolation
uv run python -m asr_eval.bench.check qwen2-audio-vad
# produces bad output

# faster-whisper
uv sync --extra models_stable --extra datasets
uv pip install faster_whisper
# insert block about LD_LIBRARY_PATH
uv run python -m asr_eval.bench.check faster-whisper-internal-vad --audio EN_LONG

# yandex speechkit
uv sync --extra models_stable --extra datasets
uv pip install yandex-speechkit
uv run python -m asr_eval.bench.check yandex-speechkit --audio EN_LONG
#  need api key

# salute speech
uv sync --extra models_stable --extra datasets
uv pip install salute_speech
uv run python -m asr_eval.bench.check salute-api --audio EN_LONG
#  need api key