VENVS_DIR=${VENVS_DIR:=/home/user/.venvs}
STORAGE=${STORAGE:=/asr/audiobench_runs}
CACHE=${CACHE:=/asr/dashboard_cache}
STREAMING_EVALS=${STREAMING_EVALS:=/asr/streaming_evals}

#############################################

DATASETS=" \
multivariant-v2:p=default:n=all! \
sova-rudevices-multivariant:p=default:n=all! \
audioset-nonspeech:p=default,ru-norm:n=1100! \
fleurs-ru:p=default,ru-norm:n=500! \
golos-farfield:p=default,ru-norm:n=1100! \
golos-crowd:p=default,ru-norm:n=1350! \
podlodka:p=default,ru-norm:n=all! \
rulibrispeech:p=default,ru-norm:n=all! \
"

DATASETS_HIGH_NOISE=" \
multivariant-v2:p=default:a=high-noise:n=all! \
sova-rudevices-multivariant:p=default:a=high-noise:n=all! \
audioset-nonspeech:p=default,ru-norm:a=high-noise:n=500! \
fleurs-ru:p=default,ru-norm:a=high-noise:n=500! \
golos-farfield:p=default,ru-norm:a=high-noise:n=500! \
golos-crowd:p=default,ru-norm:a=high-noise:n=500! \
podlodka:p=default,ru-norm:a=high-noise:n=all! \
rulibrispeech:p=default,ru-norm:a=high-noise:n=500! \
"

DATASETS_MEDIUM_NOISE=" \
multivariant-v2:p=default:a=medium-noise:n=all! \
sova-rudevices-multivariant:p=default:a=medium-noise:n=all! \
audioset-nonspeech:p=default,ru-norm:a=medium-noise:n=500! \
fleurs-ru:p=default,ru-norm:a=medium-noise:n=500! \
golos-farfield:p=default,ru-norm:a=medium-noise:n=500! \
golos-crowd:p=default,ru-norm:a=medium-noise:n=500! \
podlodka:p=default,ru-norm:a=medium-noise:n=all! \
rulibrispeech:p=default,ru-norm:a=medium-noise:n=500! \
"

DATASETS_SMALLBENCH=" \
multivariant-v2:p=default:n=all! \
sova-rudevices-multivariant:p=default:n=all! \
fleurs-ru:p=ru-norm:n=500! \
golos-crowd:p=ru-norm:n=500! \
"

DATASETS_STREAMING=" \
sova-rudevices-multivariant:p=default:n=all! \
audioset-nonspeech:p=default,ru-norm:n=1100! \
fleurs-ru:p=default,ru-norm:n=500! \
golos-farfield:p=default,ru-norm:n=1100! \
golos-crowd:p=default,ru-norm:n=1350! \
rulibrispeech:p=default,ru-norm:n=all! \
"

DATASETS_SMALLBENCH_STREAMING=" \
sova-rudevices-multivariant:p=default:n=100! \
fleurs-ru:p=ru-norm:n=100! \
golos-crowd:p=ru-norm:n=100! \
"

#############################################

WHISPER_PIPELINES=" \
wav2vec2-large-xlsr-53-russian \
wav2vec2-xls-r-1b-russian \
wav2vec2-large-ru-golos \
wav2vec2-large-ru-golos-lm-vosk-0.42 \
wav2vec2-large-ru-golos-lm-t-one \
wav2vec2-large-ru-golos-lm-bond005 \
whisper-large-v3-turbo \
whisper-podlodka-turbo \
whisper-large-v3 \
vikhr-borealis-vad \
"

GIGAAM_PIPELINES=" \
gigaam3-ctc-e2e \
gigaam3-rnnt-e2e-vad \
gigaam2-ctc \
gigaam2-ctc-blank \
gigaam2-ctc-fp16 \
gigaam3-ctc \
gigaam3-ctc-fp16 \
gigaam2-ctc-vad \
gigaam2-rnnt-vad \
gigaam3-rnnt-vad \
gigaam2-ctc-lm-vosk-0.42 \
gigaam2-ctc-lm-t-one \
gigaam2-ctc-lm-bond005 \
gigaam3-ctc-lm-t-one \
"

NEMO_PIPELINES=" \
canary-1b-v2-vad \
parakeet-tdt-0.6b-v3-vad \
stt-ru-fastconformer-vad \
stt-ru-conformer-ctc-large-vad \
"

VOSK_PIPELINES=" \
vosk-0.54-vad \
"

TONE_PIPELINES=" \
t-one-vad \
"

OUR_PRIVATE_MODEL=" \
our-pipeline-name \
"

#############################################

if [[ $1 == run || $1 == run_default ]]; then
    $VENVS_DIR/ours_v1.3.4/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $OUR_PRIVATE_MODEL --suffix='-v1.3.4'
    $VENVS_DIR/ours_v1.3.5/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $OUR_PRIVATE_MODEL --suffix='-v1.3.5'
    $VENVS_DIR/whisper/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $WHISPER_PIPELINES
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $GIGAAM_PIPELINES
    $VENVS_DIR/nemo/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $NEMO_PIPELINES
    $VENVS_DIR/vosk/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $VOSK_PIPELINES
    $VENVS_DIR/tone/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS -p $TONE_PIPELINES
fi

#############################################

DATASETS_CHECK=" \
multivariant-v2:n=all! \
sova-rudevices-multivariant:n=all! \
audioset-nonspeech:n=100! \
"

if [[ $1 == run_check ]]; then
    $VENVS_DIR/ours_v1.3.4/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_CHECK -p $OUR_PRIVATE_MODEL --suffix='-v1.3.4'
    $VENVS_DIR/ours_v1.3.5/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_CHECK -p $OUR_PRIVATE_MODEL --suffix='-v1.3.5'
fi

if [[ $1 == dash_check ]]; then
    $VENVS_DIR/asr_eval/bin/python -m asr_eval.bench.dashboard.run -s $STORAGE -c $CACHE \
        -d $DATASETS_CHECK -p $CHECK_PIPELINES
fi

#############################################

WHISPER_STREAMING_PIPELINES=" \
whisper-podlodka-turbo-quasi-streaming \
"

GIGAAM_STREAMING_PIPELINES=" \
gigaam2-ctc-quasi-streaming \
"

VOSK_STREAMING_PIPELINES=" \
vosk-ru-0.42-streaming \
"

TONE_STREAMING_PIPELINES=" \
t-one-streaming \
"

NEMO_STREAMING_PIPELINES=" \
stt-ru-conformer-ctc-large-vad-quasi-streaming \
"

if [[ $1 == run || $1 == run_streaming ]]; then
    $VENVS_DIR/whisper/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_STREAMING -p $WHISPER_STREAMING_PIPELINES
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_STREAMING -p $GIGAAM_STREAMING_PIPELINES
    $VENVS_DIR/vosk/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_STREAMING -p $VOSK_STREAMING_PIPELINES
    $VENVS_DIR/tone/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_STREAMING -p $TONE_STREAMING_PIPELINES
    $VENVS_DIR/nemo/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_STREAMING -p $NEMO_STREAMING_PIPELINES
fi

if [[ $1 == run || $1 == run_streaming_small ]]; then
    $VENVS_DIR/whisper/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_SMALLBENCH_STREAMING -p $WHISPER_STREAMING_PIPELINES
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_SMALLBENCH_STREAMING -p $GIGAAM_STREAMING_PIPELINES
    $VENVS_DIR/vosk/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_SMALLBENCH_STREAMING -p $VOSK_STREAMING_PIPELINES
    $VENVS_DIR/tone/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_SMALLBENCH_STREAMING -p $TONE_STREAMING_PIPELINES
    $VENVS_DIR/nemo/bin/python -m asr_eval.bench.run -s $STORAGE -d $DATASETS_SMALLBENCH_STREAMING -p $NEMO_STREAMING_PIPELINES
fi

if [[ $1 == streaming_evals ]]; then
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.streaming_evals -s $STORAGE -o $STREAMING_EVALS
fi

if [[ $1 == streaming_dash ]]; then
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.dashboard.run -s $STORAGE -c $CACHE -d $DATASETS_STREAMING
fi

if [[ $1 == small_streaming_dash ]]; then
    $VENVS_DIR/gigaam/bin/python -m asr_eval.bench.dashboard.run -s $STORAGE -c $CACHE -d $DATASETS_SMALLBENCH_STREAMING
fi

#############################################

if [[ $1 == dash ]]; then
    $VENVS_DIR/asr_eval/bin/python -m asr_eval.bench.dashboard.run -s $STORAGE -c $CACHE \
        -d $DATASETS $DATASETS_HIGH_NOISE $DATASETS_MEDIUM_NOISE
fi

if [[ $1 == dash_mini ]]; then
    $VENVS_DIR/asr_eval/bin/python -m asr_eval.bench.dashboard.run -s $STORAGE -c $CACHE \
        -d $DATASETS_SMALLBENCH
fi