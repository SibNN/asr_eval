# run this script from the asr_eval dir

make_and_activate_venv() {
    uv venv $1 --python 3.12 --allow-existing
    source $1/bin/activate
    uv pip install -e $2 --config-settings editable_mode=strict
}

if command -v uv >/dev/null 2>&1; then
    PIP="uv pip"
else
    PIP="pip"
fi

# dev venv (without most of the models, for running dashboard etc.)
make_and_activate_venv .venvs/asr_eval .[datasets,dev,normalize]
$PIP install "torch==2.8.*" "torchaudio==2.8.*" torchcodec==0.7 transformers
$PIP install git+https://github.com/salute-developers/GigaAM
$PIP install -r installation/all_nodeps.txt --no-deps

# Whisper, wav2vec2 + KenLM decoding, Vikhr Borealis
make_and_activate_venv .venvs/whisper .[datasets]
$PIP install -r ./installation/whisper.txt
bash ./installation/kenlm.sh  # for KenLM support
bash ./installation/vosk_lm.sh  # for Vosk KenLM support

# GigaAM
make_and_activate_venv .venvs/gigaam .[datasets]
$PIP install -r ./installation/gigaam.txt
bash ./installation/kenlm.sh  # for KenLM support

# Vosk
make_and_activate_venv .venvs/vosk .[datasets]
cmake --version || (echo "Cmake not found; run: sudo apt install cmake -y"; exit)
$PIP install -r ./installation/vosk.txt

# T-One
make_and_activate_venv .venvs/tone .[datasets]
bash ./installation/tone.sh

# Voxtral via VLLM
make_and_activate_venv .venvs/voxtral .[datasets]
$PIP install vllm[audio]

# Nemo
make_and_activate_venv .venvs/nemo .[datasets]
$PIP install -r ./installation/nemo.txt

# Speechbrain
make_and_activate_venv .venvs/speechbrain .[datasets]
$PIP install -r ./installation/speechbrain.txt

# Flamingo
not working
make_and_activate_venv .venvs/flamingo .[datasets]
$PIP install -r ./installation/flamingo.txt

# Gemma3n
make_and_activate_venv .venvs/gemma3n .[datasets]
$PIP install -r ./installation/gemma3n.txt

# Qwen2-Audio
make_and_activate_venv .venvs/qwen2audio .[datasets]
$PIP install -r ./installation/qwen2audio.txt
$PIP install flash-attn --no-build-isolation

# Faster-Whisper
make_and_activate_venv .venvs/faster_whisper .[datasets]
$PIP install faster_whisper

# Yandex-speechkit API and Salute API
make_and_activate_venv .venvs/api .[datasets]
$PIP install yandex-speechkit
$PIP install salute_speech