import json
from typing import TypedDict
import wave

import pytest
import librosa
import numpy as np
import numpy.typing as npt
from vosk import Model, KaldiRecognizer # type: ignore

from asr_eval.streaming.model import PartialTranscription
from asr_eval.streaming.caller import receive_full_transcription
from asr_eval.streaming.models.vosk import VoskStreaming
from asr_eval.streaming.sender import StreamingAudioSender

@pytest.mark.parametrize('frames_per_chunk, prediction', [
    (4000, ['one zero zero zero one', 'nah no to i know', 'zero one eight zero three']),
    (64000, ['one zero zero zero one', 'nah no to i know zero one eight zero three']),
])
def test_vosk_KaldiRecognizer(frames_per_chunk: int, prediction: list[str]):
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vosk_wrapper():
    waveform: npt.NDArray[np.floating]
    waveform, _ = librosa.load('tests/testdata/vosk.wav', sr=16_000) # type: ignore
    waveform_bytes = np.int16(waveform * 32768).tobytes()

    asr = VoskStreaming()
    asr.start_thread()
    
    class Sample(TypedDict):
        input: StreamingAudioSender
        output: str
    
    samples: list[Sample] = [
        {
            'input': StreamingAudioSender(id=0, audio=waveform_bytes, speed_multiplier=5),
            'output': 'one zero zero zero one nah no to i know zero one eight zero three',
        },
        {
            'input': StreamingAudioSender(id=1, audio=waveform_bytes, speed_multiplier=100),
            'output': 'one zero zero zero one nah no to i know zero one eight zero three',
        },
    ]

    for sample in samples:
        sample['input'].start_sending(send_to=asr.input_buffer)
            
    for sample in samples:
        chunks = receive_full_transcription(asr=asr, id=sample['input'].id)
        assert PartialTranscription.join(chunks) == sample['output']

    for sample in samples:
        sample['input'].join()
    asr.stop_thread()


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")
def test_vosk54_wrapper():
    from asr_eval.models.vosk import VoskV54
    
    waveform: npt.NDArray[np.float64]
    waveform, _ = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000) # type: ignore

    model = VoskV54()
    texts = model.transcribe([waveform])
    assert texts == ['и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно']

    model = VoskV54(device='cuda')
    texts = model.transcribe([waveform])
    assert texts == ['и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно']