rm -rf docs/build/*
python -m sphinx.ext.apidoc -o docs/source \
    -H asr_eval --no-toc --no-headings --force asr_eval/
python -m asr_eval.utils.autodoc
python -m sphinx.cmd.build -b html docs/source docs/build