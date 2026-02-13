uv pip install .[dev,normalize,dash] "datasets<4" \
  "torch==2.8.*" "torchaudio==2.8.*" torchcodec==0.7 \
  "transformers<5" "pyannote.audio>=4" "lightning<2.6" \
  git+https://github.com/salute-developers/GigaAM
uv pip install -r installation/all_nodeps.txt --no-deps
python -m pyright -p pyrightconfig.json asr_eval
python -m pytest tests
python -m xdoctest asr_eval
sphinx-build -M doctest docs/source docs/build docs/source/guide_alignment_wer.rst
sphinx-build -M doctest docs/source docs/build docs/source/guide_streaming_evaluation.rst
sphinx-build -M doctest docs/source docs/build docs/source/guide_evaluation_dashboard.rst
rm -rf docs/build
sphinx-build docs/source docs/build

uv pip install sphinx-simplepdf
cd docs
make simplepdf

