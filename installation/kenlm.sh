# numpy 2.x is not supported by pyctcdecode, but
# it works and nothing will break

if command -v uv >/dev/null 2>&1; then
    PIP="uv pip"
else
    PIP="pip"
fi

$PIP install pyctcdecode==0.5.0 --no-deps
$PIP install kenlm "pygtrie>=2.1,<3.0"  # for pyctcdecode