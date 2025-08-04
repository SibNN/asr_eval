Evaluation tools for Automatic Speech Recognition

# Installation

**Prerequisites**

```
sudo apt install python3-dev python3.12-dev python3.12-venv ffmpeg
python3.12 -m venv venv
source venv/bin/activate
```

python3-dev is required to build some requirements; FFMpeg is needed for GigaAM.

For GPU support:

```
sudo ubuntu-drivers install
sudo reboot
sudo apt install nvidia-cuda-toolkit
```

If you see torchcodec errors, install a specific version from the compatibility table:
https://github.com/pytorch/torchcodec#installing-cpu-only-torchcodec

**Basic installation**

```
pip install -e .
```

**Full installation**

```
pip install -e .[all]
```

This will install all the required optional dependencies listed below. Note that this will not install Vosk 0.54, Pisets and T-One since they are not on PyPI.

**Dev installation**

Instal dev version to run tests and build docs:

```
pip install -e .[dev]
```

**Speechbrain support**

```
pip install -e .[speechbrain]
```

**NVIDIA conformer support**

```
pip install -e .[nvidia-conformer]
```

**Pisets support**

Locate to the Pisets dir and run:

```
pip install -e .
```

**Vosk support**

```
pip install -e .[vosk]
```

If you need Vosk 0.54, it requires packages that are not on PyPI, install them manually:

```
pip install sentencepiece torch==2.5.1 huggingface_hub
pip install kaldifeat==1.25.5.dev20250203+cuda12.4.torch2.5.1 -f https://csukuangfj.github.io/kaldifeat/cuda.html
pip install k2==1.24.4.dev20250208+cuda12.4.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
pip install git+https://github.com/lhotse-speech/lhotse
sudo apt install cmake
pip install git+https://github.com/k2-fsa/icefall
```

**T-One support**

```
pip install git+https://github.com/voicekit-team/T-one
```

**Qwen support**

```
pip install -e .[qwen-audio]
```

Additionally to speed up inference you can run

```
pip install flash-attn --no-build-isolation
```