Installation
###################

Since *asr_eval* provides wrappers for many ASR models, they bring
many heavy dependencies. Moreover, some models have *incompatible dependencies*.
Thus, we provide several installation types:

1. :ref:`Typical installation`, **suitable for most cases**.
2. :ref:`Lightweight installation` without optional dependencies.
3. :ref:`Model installations` for specific ASR models.
4. :ref:`Dev installation` for contributing, to type-check the whole project.

.. _Typical installation:

Typical installation
********************

*asr_eval* requires Python 3.12. A relatively complete installation that
enables the core components, many models and datasets is done this way:

.. code-block:: bash

    sudo apt install python3-dev python3.12-dev python3.12-venv ffmpeg
    pip install asr_eval[normalize,dash] "datasets<4" \
        "torch==2.8.*" "torchaudio==2.8.*" torchcodec==0.7 \
        "transformers<5" "pyannote.audio>=4" "lightning<2.6" \
        git+https://github.com/salute-developers/GigaAM

Notes for some optional packages:

.. list-table:: 
   :widths: 2 10
   :header-rows: 0

   * - **ffmpeg**
     - Is required for audio datasets to load.
   * - **datasets**
     - Is required for audio datasets to load.
       You can install :code:`datasets>=4`, but some of the datasets won't load,
       becuse they do not work without :code:`trust_remote_code=True`.
   * - **torch**
     - Is required for most of the models to work.
       Newer versions of :code:`torch`, :code:`torchaudio` and :code:`torchcodec`
       may also work (:code:`torchaudio` is in active refactoring state now, and for
       :code:`torchcodec` you need to manually select a version from the compatibility table).
       :code:`torchaudio` is required to run the :code:`forced_alignment` utility and derivative
       operations, such as filling timings for streaming evaluation. If you omit :code:`torchaudio`,
       you may therefore drop a heavy :code:`torch` dependency, but :code:`ImportError` may occur for some
       utilities.
   * - **transformers**
     - Enables many model wrappers such as Whisper, wav2vec2 and Vikhr Borealis.
       If not installed, :code:`ImportError` will occur when trying to instantiate these wrappers.
   * - **gigaam**
     - Enables GigaAM model, that helps to fill word timings for Russian streaming evaluiation.
   * - **pyannote.audio**
     - Is required for Pyannote voice activity detector, which acts as a segmenter
       in many speech recognition pipelines for audios longer than 30 seconds. If not installed,
       :code:`ImportError` will occur when trying to instantiate a Pyannote VAD. Also requires
       :code:`lightning<2.6` to work correctly.
       

Let's validate the installation:

.. code-block:: bash

    python -c "from asr_eval.bench.datasets import get_dataset; \
        print(f'{len(get_dataset('podlodka'))=}')"
    python -m asr_eval.bench.run -s whisper-predictions -p whisper-tiny -d fleurs-ru:n=5
    python -m asr_eval.bench.dashboard.run -s whisper-predictions

The first command should output: :code:`len(get_dataset('podlodka'))=20`.
The next commands download fleurs-ru dataset, run whisper-tiny on the first 5
samples, saves to the :code:`whisper-predictions` dir and start a web dashboard.

.. _Lightweight installation:

Lightweight installation
****************************

Run :code:`pip install asr_eval` to install only the core components
(no models or datasets support).

.. _Dev installation:

Dev installation
********************

Dev installation allows to run type checker, tests, and build docs. Since models
have conflucting dependencies, a solution is to install model packages without
dependencies. This will fail if calling these models at runtime, but is fine
for type checker.

Dev installation in editable mode is done as follows (you can add packages
from :ref:`Typical installation` if you need to run the corresponding components):

.. code-block:: bash

    git clone https://github.com/SibNN/asr_eval
    cd asr_eval
    pip install -e .[dev]
    pip install -r installation/all_nodeps.txt --no-dependencies

Run type checker, tests, doctests:

.. code-block:: bash

    python -m pyright -p pyrightconfig.json asr_eval
    python -m pytest tests
    python -m xdoctest asr_eval

Build docs:

.. code-block:: bash

    python -m sphinx.ext.apidoc -o docs/source \
        -H asr_eval --no-toc --no-headings --force asr_eval/
    python -m asr_eval.utils.autodoc
    python -m sphinx.cmd.build -b html docs/source docs/build

Run CI/CD (local):

.. code-block:: bash

    apt install gh
    gh extension install https://github.com/nektos/gh-act
    gh act -v --artifact-server-path /tmp/artifacts

.. _Model installations:

Model installations
***********************

A model installation is a set of addidional requirements to make certain models
and pipelines work. It is not possible to install all the models at once, because
some of them have incompatible dependencies. It is recommended to create a separate
venv for each installation.

Standard steps:

1. Install :code:`asr_eval "datasets<4"` to support Hugging Face datasets.
2. Run a specific installation script from the :code:`asr_eval/installation` dir
   in :code:`https://github.com/SibNN/asr_eval`.

.. admonition:: Note

    We recommend to use **UV** since it is much faster and takes less disk space. To use UV,
    install it with :code:`wget -qO- https://astral.sh/uv/install.sh | sh` and then
    use :code:`uv pip install` instead of :code:`pip install`. If using
    :code:`--no-dependencies` flag, replace it with :code:`--no-deps` when using UV.

.. admonition:: Note

    :code:`pyannote-audio` is included for many models, because in *asr_eval* this
    is a default segmenter for long audios. To run models based on Pyannote VAD
    you will need to specify HF_TOKEN (get it at https://huggingface.co/settings/tokens).

If you want to make all the environments with one command, refer to the section
:ref:`Full multi-enviromnent installation`. Otherwise, see the next sections.


Whisper, wav2vec2, Vikhr Borealis
======================================

.. code-block:: bash

    # install
    pip install -r ./installation/whisper.txt
    bash ./installation/kenlm.sh  # for KenLM support
    # test
    python -m asr_eval.bench.check whisper-large-v3
    python -m asr_eval.bench.check wav2vec2-large-xlsr-53-russian
    python -m asr_eval.bench.check vikhr-borealis-vad

Additional steps if you need wav2vec2 pipelines with Vosk kenLM:

.. code-block:: bash

    # install
    bash ./installation/vosk_lm.sh
    # test (will load a model for ~1min)
    python -m asr_eval.bench.check wav2vec2-large-ru-golos-lm-vosk-0.42

GigaAM
======================================

.. code-block:: bash

    # install
    pip install -r ./installation/gigaam.txt
    bash ./installation/kenlm.sh  # for KenLM support
    # test
    python -m asr_eval.bench.check gigaam-rnnt-vad
    python -m asr_eval.bench.check gigaam-ctc-lm-t-one

Additional steps if you need pipelines with Vosk ru kenLM:

.. code-block:: bash

    # install
    bash ./installation/vosk_lm.sh
    # test (will load a model for ~1min)
    python -m asr_eval.bench.check gigaam-ctc-lm-vosk-0.42

Vosk
======================================

.. code-block:: bash

    # install
    sudo apt install cmake -y
    pip install -r ./installation/vosk.txt
    # test
    python -m asr_eval.bench.check vosk-0.54-vad
    python -m asr_eval.bench.check vosk-ru-0.42-streaming

.. _t_one_installation:

T-One
======================================

.. code-block:: bash

    # install
    bash ./installation/tone.sh
    # test
    python -m asr_eval.bench.check t-one-vad

Voxtral via VLLM
======================================

VLLM is not terminated, TODO fix

.. code-block:: bash

    # install
    pip install vllm[audio]
    # test
    python -m asr_eval.bench.check voxtral-3B

Nemo
======================================

.. code-block:: bash

    # install
    pip install -r ./installation/nemo.txt
    # test
    python -m asr_eval.bench.check canary-1b-v2-vad

Speechbrain
======================================

.. code-block:: bash

    # install
    pip install -r ./installation/speechbrain.txt
    # test
    python -m asr_eval.bench.check speechbrain-conformer-gigaspeech-streaming

Flamingo
======================================

Not working anymore, TODO fix

.. code-block:: bash

    # install
    pip install -r ./installation/flamingo.txt
    # test
    python -m asr_eval.bench.check flamingo-ru-vad

Gemma3n
======================================

Too slow currently, TODO fix

.. code-block:: bash

    # install
    pip install -r ./installation/gemma3n.txt
    # test
    python -m asr_eval.bench.check gemma3n-ru-vad

Qwen2-Audio
======================================

Produces bad output, TODO fix

.. code-block:: bash

    # install
    pip install -r ./installation/qwen2audio.txt
    pip install flash-attn --no-build-isolation
    # test
    python -m asr_eval.bench.check qwen2-audio-vad

Faster-Whisper
======================================

.. code-block:: bash

    # install
    pip install faster_whisper
    # test
    python -m asr_eval.bench.check faster-whisper-internal-vad

If it says "Unable to load any of {libcudnn_ops.so.9.1.0, ...}" - then find this file:

.. code-block:: bash
    
    sudo find / -name "libcudnn_ops.so*" 2>/dev/null

And add the directory containing this file to :code:`LD_LIBRARY_PATH`, for example:

.. code-block:: bash
    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:..../lib/python3.12/site-packages/nvidia/cudnn/lib

Yandex-speechkit API
======================================

.. code-block:: bash

    # install
    pip install yandex-speechkit
    # test
    python -m asr_eval.bench.check yandex-speechkit

Salute API
======================================

.. code-block:: bash

    # install
    pip install salute_speech
    # test
    python -m asr_eval.bench.check salute-api

.. _Full multi-enviromnent installation:

Full multi-enviromnent installation
****************************************

The automation script below creates :code:`.venvs` dir and makes several enviromnents:
:code:`.venvs/asr_eval` for dev installation and others for model installations. All
the created venvs contain editable installation for asr_eval.

.. code-block:: bash

    git clone https://github.com/SibNN/asr_eval
    cd asr_eval
    bash installation/everything.sh

After :code:`everything.sh` finishes, you can run all the required tests, such as:

.. code-block:: bash

    # run asr_eval basic tests
     .venvs/venv_asr_eval/bin/python -m pytest tests
    # test a specific model
    .venvs/venv_tone/bin/python -m asr_eval.bench.check t-one-vad