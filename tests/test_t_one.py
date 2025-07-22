from typing import TypedDict
import librosa
import numpy as np

import pytest

from asr_eval.models.t_one_wrapper import TOneStreaming
from asr_eval.streaming.model import TranscriptionChunk
from asr_eval.streaming.caller import receive_full_transcription
from asr_eval.streaming.sender import StreamingAudioSender
from asr_eval.utils.types import FLOATS

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_t_one():
    waveform: FLOATS = librosa.load('tests/testdata/formula1.mp3', sr=8_000)[0] # type: ignore
    waveform_ints = (waveform * 32768).astype(np.int16)

    asr = TOneStreaming()
    asr.start_thread()
    
    class Sample(TypedDict):
        input: StreamingAudioSender
        output: str

    samples: list[Sample] = [
        {
            'input': StreamingAudioSender(id=0, audio=waveform_ints, speed_multiplier=5),
            'output': (
                'седьмого восьмого мая в пуэрто рико прошел шестнадцатый этап формулы один'
                ' с фондом сто тысяч долларов победителем стал гонщик мерседеса джордж рассел'
            ),
        },
    ]

    for sample in samples:
        sample['input'].start_sending(send_to=asr.input_buffer)
            
    for sample in samples:
        chunks = receive_full_transcription(asr=asr, id=sample['input'].id)
        assert TranscriptionChunk.join(chunks) == sample['output']

    for sample in samples:
        sample['input'].join()
    asr.stop_thread()