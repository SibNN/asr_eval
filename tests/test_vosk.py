import json
import wave
from collections import defaultdict

import pytest
import librosa
import numpy as np
from vosk import Model, KaldiRecognizer

from asr_eval.streaming.model import RECORDING_ID_TYPE, Signal
from asr_eval.streaming.models.vosk import VoskStreaming
from asr_eval.streaming.sender import StreamingAudioSender
from asr_eval.streaming.transcription import PartialTranscription

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vosk_wrapper():
    waveform, _ = librosa.load('tests/testdata/vosk.wav', sr=16_000)
    waveform_bytes = np.int16(waveform * 32768).tobytes()

    asr = VoskStreaming()
    asr.start_thread()

    sender = StreamingAudioSender(id=0, audio=waveform_bytes, speed_multiplier=5, send_to=asr.input_buffer)
    sender.start_sending()

    results: dict[RECORDING_ID_TYPE, list[PartialTranscription]] = defaultdict(list)
    finished: dict[RECORDING_ID_TYPE, bool] = {0: False}

    while True:
        id, output = asr.output_buffer.get()
        if output is Signal.FINISH:
            finished[id] = True
            if all(finished.values()):
                break
        else:
            results[id].append(output)

    sender.join()
    asr.stop_thread()