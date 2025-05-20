from typing import TypedDict
import numpy as np
import pytest

from asr_eval.streaming.caller import wait_for_transcribing
from asr_eval.streaming.model import DummyASR
from asr_eval.streaming.sender import StreamingAudioSender


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_duplicate_input_ids():
    """Erroneously send the same recording ID two times"""
    asr = DummyASR()
    asr.start_thread()
    for _ in range(2):
        StreamingAudioSender(id=0, audio=np.zeros(16_000), real_time_interval_sec=1, send_to=asr.input_buffer).start_sending()
    with pytest.raises(RuntimeError):
        for _ in range(4):
            asr.output_buffer.get()


def test_basic():  # TODO extend test suite
    asr = DummyASR()
    asr.start_thread()
    
    class Sample(TypedDict):
        input: StreamingAudioSender
        output: list[str]
    
    samples: list[Sample] = [
        {
            'input': StreamingAudioSender(id=0, audio=np.zeros(16_000 * 5), speed_multiplier=27, send_to=asr.input_buffer),
            'output': [str(x) for x in range(5)]
        },
        {
            'input': StreamingAudioSender(id=1, audio=np.zeros(16_000 * 10), speed_multiplier=20, send_to=asr.input_buffer),
            'output': [str(x) for x in range(10)]
        },
    ]
    
    for sample in samples:
        sample['input'].start_sending()
    
    results = wait_for_transcribing(asr, ids=[sample['input'].id for sample in samples])
            
    for sample in samples:
        assert [x.text for x in results[sample['input'].id]] == sample['output']
    
    for sample in samples:
        sample['input'].join()
    asr.stop_thread()