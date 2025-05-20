import json
import wave

import pytest
import librosa
import numpy as np
import numpy.typing as npt
from vosk import Model, KaldiRecognizer # type: ignore

from asr_eval.streaming.caller import wait_for_transcribing
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
        if rec.AcceptWaveform(data): # type: ignore
            transcription_done.append(json.loads(rec.Result())['text']) # type: ignore
            transcription_partial = None
        else:
            transcription_partial = json.loads(rec.PartialResult())['partial'] # type: ignore

    if transcription_partial:
        transcription_done.append(transcription_partial)

    assert transcription_done == ['one zero zero zero one', 'nah no to i know', 'zero one eight zero three']


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vosk_wrapper():
    waveform: npt.NDArray[np.float64]
    waveform, _ = librosa.load('tests/testdata/vosk.wav', sr=16_000) # type: ignore
    waveform_bytes = np.int16(waveform * 32768).tobytes()

    asr = VoskStreaming()
    asr.start_thread()

    sender = StreamingAudioSender(id=0, audio=waveform_bytes, speed_multiplier=5, send_to=asr.input_buffer)
    sender.start_sending()
    
    results = wait_for_transcribing(asr, ids=[0])
    assert PartialTranscription.join(results[0]) == 'one zero zero zero one nah no to i know zero one eight zero three'

    sender.join()
    asr.stop_thread()