Installation
###################

Prerequisites
*****************

Basic installation:

.. code-block:: bash

    sudo apt install python3-dev python3.12-dev python3.12-venv ffmpeg
    python3.12 -m venv venv
    source venv/bin/activate

    pip install -e .[dev]

A package python3-dev is required to build some requirements; FFMpeg is needed for GigaAM. For GPU support:

.. code-block:: bash

    sudo ubuntu-drivers install
    sudo reboot
    sudo apt install nvidia-cuda-toolkit

If you see torchcodec errors, install a specific version from the `compatibility table <https://github.com/pytorch/torchcodec#installing-cpu-only-torchcodec>`_:

Building docs
*****************

.. code-block:: bash

    sphinx-apidoc -o docs/source -H asr_eval -V 0.0.1 --no-toc --no-headings --force asr_eval/
    python -m asr_eval.utils.autodoc
    sphinx-build -b html docs/source docs/build

Preparing models
*****************

It is better to prepare a separate Python environment for each model, bacause they ahve conflicting dependencies.

Whisper
------------------

.. code-block:: bash

    python3.12 -m venv tmp/venv_whisper
    tmp/venv_whisper/bin/python -m pip install -e .
    tmp/venv_whisper/bin/python -m pip install transformers
    tmp/venv_whisper/bin/python -m asr_eval.bench.check whisper-large-v3

GigaAM
------------------

.. code-block:: bash
    
    python3.12 -m venv tmp/venv_gigaam
    tmp/venv_gigaam/bin/python -m pip install -e .
    tmp/venv_gigaam/bin/python -m pip install gigaam pyannote.audio
    tmp/venv_gigaam/bin/python -m pip install onnxruntime==1.22.1
    tmp/venv_gigaam/bin/python -m asr_eval.bench.check gigaam-rnnt-vad

Vosk 0.54 and streaming
----------------------------

.. code-block:: bash
    
    sudo apt install cmake
    python3.12 -m venv tmp/venv_vosk
    tmp/venv_vosk/bin/python -m pip install -e .
    tmp/venv_vosk/bin/python -m pip install vosk pyannote.audio sentencepiece torch==2.5.1 huggingface_hub
    tmp/venv_vosk/bin/python -m pip install gigaam --no-dependencies  # for segmenting
    tmp/venv_vosk/bin/python -m pip install kaldifeat==1.25.5.dev20250203+cuda12.4.torch2.5.1 -f https://csukuangfj.github.io/kaldifeat/cuda.html
    tmp/venv_vosk/bin/python -m pip install k2==1.24.4.dev20250208+cuda12.4.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
    tmp/venv_vosk/bin/python -m pip install git+https://github.com/lhotse-speech/lhotse
    tmp/venv_vosk/bin/python -m pip install git+https://github.com/k2-fsa/icefall
    tmp/venv_vosk/bin/python -m asr_eval.bench.check vosk-0.54-vad

Flamingo
----------------------------

.. code-block:: bash
    
    python3.12 -m venv tmp/venv_flamingo
    tmp/venv_flamingo/bin/python -m pip install -e .
    tmp/venv_flamingo/bin/python -m pip install numpy==1.26.4 whisper accelerate==0.34.2 pytorchvideo==0.1.5 torchvision deepspeed==0.15.4 transformers==4.46.0 pyannote.audio opencv-python-headless==4.8.0.76 kaldiio loguru
    tmp/venv_flamingo/bin/python -m pip install gigaam --no-dependencies  # for segmenting
    tmp/venv_flamingo/bin/python -m asr_eval.bench.check flamingo-ru-vad

Gemma3n
----------------------------

.. code-block:: bash
    
    export HF_TOKEN=...  # your token
    python3.12 -m venv tmp/venv_gemma3n
    tmp/venv_gemma3n/bin/python -m pip install -e .
    tmp/venv_gemma3n/bin/python -m pip install "transformers==4.54.1" pyannote.audio torchvision accelerate timm
    tmp/venv_gemma3n/bin/python -m pip install gigaam --no-dependencies  # for segmenting
    tmp/venv_gemma3n/bin/python -m asr_eval.bench.check gemma3n-ru-vad

Pisets and Pisets-legacy
----------------------------

.. code-block:: bash
    
    PISETS_DIR=...
    ASR_EVAL_DIR=...
    python3.12 -m venv tmp/venv_pisets
    tmp/venv_pisets/bin/python -m pip install -e .
    tmp/venv_pisets/bin/python -m pip install transformers
    git clone https://github.com/bond005/pisets tmp/pisets_legacy
    cd $PISETS_DIR
    $ASR_EVAL_DIR/tmp/venv_pisets/bin/python -m pip install .
    cd $ASR_EVAL_DIR
    tmp/venv_pisets/bin/python -m asr_eval.bench.check pisets-legacy
    tmp/venv_pisets/bin/python -m asr_eval.bench.check pisets-ru-whisper-large-v3

Qwen2-Audio
----------------------------

.. code-block:: bash
    
    python3.12 -m venv tmp/venv_qwen2audio
    tmp/venv_qwen2audio/bin/python -m pip install -e .
    tmp/venv_qwen2audio/bin/python -m pip install transformers_stream_generator "transformers>4.32.0" pyannote.audio accelerate
    tmp/venv_qwen2audio/bin/python -m pip install gigaam --no-dependencies  # for segmenting
    tmp/venv_qwen2audio/bin/python -m pip install flash-attn --no-build-isolation
    tmp/venv_qwen2audio/bin/python -m asr_eval.bench.check qwen2-audio-vad

T-One
----------------------------

.. code-block:: bash
    
    python3.12 -m venv tmp/venv_tone
    tmp/venv_tone/bin/python -m pip install -e .
    tmp/venv_tone/bin/python -m pip install pyannote.audio
    tmp/venv_tone/bin/python -m pip install gigaam --no-dependencies  # for segmenting
    tmp/venv_tone/bin/python -m pip install git+https://github.com/voicekit-team/T-one
    tmp/venv_tone/bin/python -m asr_eval.bench.check t-one-vad

Voxtral
----------------------------

.. code-block:: bash
    
    python3.12 -m venv tmp/venv_voxtral
    tmp/venv_voxtral/bin/python -m pip install -e .
    tmp/venv_voxtral/bin/python -m pip install vllm[audio] openai mistral_common
    tmp/venv_voxtral/bin/python -m asr_eval.bench.check voxtral-3B

Yandex-speechkit
----------------------------

See :code:`asr_eval.tts.yandex_speechkit.YandexSpeechKitWrapper` docstring for installation instructions.

.. code-block:: bash
    
    export YANDEX_API_KEY=...  # koyr key
    python3.12 -m venv tmp/venv_yandex_speechkit
    tmp/venv_yandex_speechkit/bin/python -m pip install -e .
    tmp/venv_yandex_speechkit/bin/python -m pip install yandex-speechkit
    tmp/venv_yandex_speechkit/bin/python -m asr_eval.bench.check yandex-speechkit