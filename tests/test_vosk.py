import json
import wave

from vosk import Model, KaldiRecognizer

def test_vosk_KaldiRecognizer():
    """
    Testing vosk.KaldiRecognizer without any wrapper code from the current framework
    """
    model = Model(lang='en-us')
    
    # need to re-create KaldiRecognizer when a new audio comes
    # https://deepwiki.com/alphacep/vosk-api/2.2-recognizer

    wf = wave.open('tests/testdata/vosk.wav', 'rb')
    rec = KaldiRecognizer(model, 16_000)

    transcription_done: list[str] = []
    transcription_partial: str | None = None

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            transcription_done.append(json.loads(rec.Result())['text'])
            transcription_partial = None
        else:
            transcription_partial = json.loads(rec.PartialResult())['partial']

    if transcription_partial:
        transcription_done.append(transcription_partial)

    assert transcription_done == ['one zero zero zero one', 'nah no to i know', 'zero one eight zero three']