DIR=~/.cache/asr_eval
if [ ! -f "$DIR/vosk-model-ru-0.42-compile/db/ru.lm.gz" ]; then
    mkdir DIR -p
    wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42-compile.zip -P $DIR
    unzip $DIR/vosk-model-ru-0.42-compile.zip -d $DIR
    rm $DIR/vosk-model-ru-0.42-compile.zip
fi

# as a result, the directory should appear:
# ~/.cache/asr_eval/vosk-model-ru-0.42-compile