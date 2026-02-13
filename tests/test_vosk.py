import json
import wave

import pytest
import librosa

from asr_eval.utils.types import FLOATS


@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.parametrize('frames_per_chunk, prediction', [
    (4000, ['one zero zero zero one', 'nah no to i know', 'zero one eight zero three']),
    (64000, ['one zero zero zero one', 'nah no to i know zero one eight zero three']),
])
def test_vosk_KaldiRecognizer(frames_per_chunk: int, prediction: list[str]):
    """
    Testing vosk.KaldiRecognizer without any wrapper code from the current framework
    """
    from vosk import Model, KaldiRecognizer # type: ignore
    
    model = Model(lang='en-us')
    
    # need to re-create KaldiRecognizer when a new audio comes
    # https://deepwiki.com/alphacep/vosk-api/2.2-recognizer

    wf = wave.open('tests/testdata/vosk.wav', 'rb')
    rec = KaldiRecognizer(model, 16_000)

    transcription_done: list[str] = []
    transcription_partial: str | None = None

    while True:
        data = wf.readframes(frames_per_chunk)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data): # type: ignore
            transcription_done.append(json.loads(rec.Result())['text']) # type: ignore
            transcription_partial = None
        else:
            transcription_partial = json.loads(rec.PartialResult())['partial'] # type: ignore

    if transcription_partial:
        transcription_done.append(transcription_partial)

    assert transcription_done == prediction


@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")
def test_vosk54_wrapper():
    from asr_eval.models.vosk54_wrapper import VoskV54
    
    waveform: FLOATS = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000)[0] # type: ignore

    model = VoskV54()
    text = model.transcribe(waveform)
    assert text == 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'

    model = VoskV54(device='cuda')
    text = model.transcribe(waveform)
    assert text == 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'