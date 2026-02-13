# T-One depends on pyctcdecode, and pyctcdecode (PyPI version) depends on numpy 1.x
# this is too restrictive, given that pyctcdecode also works perfectly for numpy 2.x

# to address this, we first install T-One and pyctcdecode without dependencies,
# then install dependencies manually

if command -v uv >/dev/null 2>&1; then
    PIP="uv pip"
else
    PIP="pip"
fi

$PIP install git+https://github.com/voicekit-team/T-one --no-deps

# installing pyctcdecode without reverting numpy to 1.x
bash ./installation/kenlm.sh

# installing t-one dependencies manually (excluding pyctcdecode) for python 3.12
# from https://github.com/voicekit-team/T-one/blob/main/pyproject.toml
# and install also pyannote VAD
$PIP install \
    "numpy (>=1.17.3,<3.0.0)" \
    "huggingface-hub (>=0.14.0,<1.0.0)" \
    "onnxruntime (>=1.12.0,<2.0.0)" \
    torch "pyannote.audio>=4" "lightning<2.6" 
