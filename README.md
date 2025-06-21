Evaluation tools for Automatic Speech Recognition

**Installation**

`pip install -e .[dev,gigaam,vosk]`

Since vosk 0.54 requires packages that are not on pypi, install them manually:

```
pip install sentencepiece torch==2.5.1 huggingface_hub
pip install kaldifeat==1.25.5.dev20250203+cuda12.4.torch2.5.1 -f https://csukuangfj.github.io/kaldifeat/cuda.html
pip install k2==1.24.4.dev20250208+cuda12.4.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
pip install git+https://github.com/lhotse-speech/lhotse
sudo apt install cmake
pip install git+https://github.com/k2-fsa/icefall
```