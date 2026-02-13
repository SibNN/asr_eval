from typing import TypedDict
import numpy as np
import pytest

from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import make_sender
from asr_eval.streaming.model import DummyASR, Signal
from asr_eval.streaming.sender import StreamingSender


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_duplicate_input_ids():
    """Erroneously send the same recording ID two times"""
    asr = DummyASR()
    asr.start_thread()
    for _ in range(2):
        _cutoffs, sender = make_sender(np.zeros(16_000), asr, uid='0')
        sender.start_sending()
    with pytest.raises(RuntimeError):
        for _ in range(4):
            asr.output_buffer.get()
    asr.stop_thread()


def test_dummy():
    asr = DummyASR()
    asr.start_thread()
    
    class Sample(TypedDict):
        sender: StreamingSender
        output: list[str]
    
    samples: list[Sample] = [
        {
            'sender': make_sender(
                np.zeros(16_000 * 5),
                asr=asr,
                speed_multiplier=27,
            )[1],
            'output': [str(x) for x in range(5)]
        },
        {
            'sender': make_sender(
                np.zeros(16_000 * 10),
                asr=asr,
                speed_multiplier=20,
            )[1],
            'output': [str(x) for x in range(10)]
        },
    ]
    
    for sample in samples:
        sample['sender'].start_sending()
    
    for sample in samples:
        chunks = list(receive_transcription(
            asr=asr, id=sample['sender'].id
        ))
        assert sample['output'] == [
            x.data.text
            for x in chunks
            if x.data is not Signal.FINISH
        ]
    
    for sample in samples:
        sample['sender'].join()
    asr.stop_thread()