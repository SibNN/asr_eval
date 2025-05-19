from collections import defaultdict
import numpy as np
import pytest

from asr_eval.streaming.model import PartialTranscription, Signal, DummyBlackBoxASR
from asr_eval.streaming.sender import RECORDING_ID_TYPE, StreamingAudioSender


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_duplicate_input_ids():
    """Erroneously send the same recording ID two times"""
    asr = DummyBlackBoxASR()
    asr.start_thread()
    for _ in range(2):
        StreamingAudioSender(id=0, audio=np.zeros(16_000), real_time_interval_sec=1, send_to=asr.input_buffer).start_sending()
    with pytest.raises(RuntimeError) as e:
        for _ in range(4):
            asr.output_buffer.get()


def test_basic():  # TODO extend test suite
    asr = DummyBlackBoxASR()
    asr.start_thread()
    
    samples = [
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

    results: dict[RECORDING_ID_TYPE, list[PartialTranscription]] = defaultdict(list)
    finished: dict[RECORDING_ID_TYPE, bool] = {sample['input'].id: False for sample in samples}

    while True:
        id, output = asr.output_buffer.get()
        if output is Signal.FINISH:
            finished[id] = True
            if all(finished.values()):
                break
        else:
            results[id].append(output)
            
    for sample in samples:
        sample['input'].join()
    
    for sample in samples:
        assert [x.word for x in results[sample['input'].id]] == sample['output']
    
    asr.join()