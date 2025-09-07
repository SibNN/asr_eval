DATASETS="-d multivariant-v1-200 common-voice-17.0 resd fleurs golos-farfield \
speech-massive rulibrispeech podlodka-full sova-rudevices youtube-lectures -m 500"

tmp/venv_whisper/bin/python -m asr_eval.bench.run -p whisper-large-v3 $DATASETS
tmp/venv_whisper/bin/python -m asr_eval.bench.run -p whisper-large-v3-turbo $DATASETS
tmp/venv_whisper/bin/python -m asr_eval.bench.run -p whisper-podlodka-turbo $DATASETS
tmp/venv_gigaam/bin/python -m asr_eval.bench.run -p gigaam-ctc $DATASETS
tmp/venv_gigaam/bin/python -m asr_eval.bench.run -p gigaam-rnnt-vad $DATASETS
tmp/venv_tone/bin/python -m asr_eval.bench.run -p t-one-vad $DATASETS
tmp/venv_vosk/bin/python -m asr_eval.bench.run -p vosk-0.54-vad $DATASETS
tmp/venv_yandex_speechkit/bin/python -m asr_eval.bench.run -p yandex-speechkit $DATASETS

tmp/venv_voxtral/bin/python -m asr_eval.bench.run -p voxtral-3B $DATASETS
tmp/venv_voxtral/bin/python -m asr_eval.bench.run -p voxtral-24B $DATASETS
tmp/venv_flamingo/bin/python -m asr_eval.bench.run -p flamingo-ru-vad $DATASETS
tmp/venv_qwen2audio/bin/python -m asr_eval.bench.run -p qwen2-audio-vad $DATASETS

tmp/venv_vosk/bin/python -m asr_eval.bench.run -p vosk-ru-0.42-offline $DATASETS

python -m asr_eval.bench.dashboard

# all datasets:
# multivariant-v1-200 200
# podlodka 20
# podlodka-full 107
# resd 280
# youtube-lectures 7
# ontico-unlabeled 170
# fleurs 775
# golos-farfield 1916
# rulibrispeech 1352
# speech-massive 2974
# sova-rudevices 5799
# common-voice-17.0 10203

# all pipelines
# whisper-large-v3
# whisper-large-v3-turbo
# whisper-podlodka-turbo
# whisper-small
# whisper-tiny
# gigaam-ctc
# gigaam-ctc-vad
# gigaam-rnnt-vad
# t-one-vad
# yandex-speechkit
# vosk-0.54-vad
# pisets-legacy
# pisets-ru-whisper-large-v3
# pisets-podlodka
# flamingo-ru-vad
# gemma3n-ru-vad
# gemma3n-ru-vad-contextual
# qwen2-audio-vad
# voxtral-3B
# voxtral-3B-mp3
# canary-1b-v2