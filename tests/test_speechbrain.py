from typing import cast, Any

import pytest
import numpy as np
import numpy.typing as npt
import librosa

from asr_eval.streaming.caller import receive_full_transcription
from asr_eval.streaming.model import TranscriptionChunk, prepare_audio_format
from asr_eval.models.speechbrain import SpeechbrainStreaming
from asr_eval.streaming.sender import StreamingAudioSender


@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::DeprecationWarning:')
def test_speechbrain_streaming():
    waveform = cast(npt.NDArray[np.floating[Any]], librosa.load('tests/testdata/vosk.wav', sr=16_000)[0]) # type: ignore
    
    asr = SpeechbrainStreaming()
    asr.start_thread()
    
    array, array_len_per_sec = prepare_audio_format(waveform, asr)
    sender = StreamingAudioSender(id=0, audio=array, speed_multiplier=5, array_len_per_sec=array_len_per_sec)
    sender.send_all_without_delays(send_to=asr.input_buffer)
    
    chunks = receive_full_transcription(asr=asr, id=sender.id)
    assert TranscriptionChunk.join(chunks) == 'ONE ZERO ZERO ZERO ONE NINE O TWO ONE O CYRIL ONE AID CYRROW THREE'
    
    asr.stop_thread()