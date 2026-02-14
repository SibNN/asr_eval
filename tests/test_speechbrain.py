import pytest

from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import make_sender
from asr_eval.streaming.model import TranscriptionChunk
from asr_eval.models.speechbrain_wrapper import SpeechbrainStreaming
from asr_eval.utils.types import FLOATS
from asr_eval.utils.audio_ops import resample


@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::DeprecationWarning:')
def test_speechbrain_streaming():
    import librosa
    
    waveform: FLOATS = librosa.load('tests/testdata/vosk.wav', sr=16_000)[0] # type: ignore
    
    asr = SpeechbrainStreaming()
    asr.start_thread()
    
    array = resample(
        waveform,
        from_sampling_rate=16_000,
        to_sampling_rate=asr.sampling_rate
    )
    _cutoffs, sender = make_sender(array, asr, speed_multiplier=5, uid='0')
    sender.start_sending(without_delays=True)
    
    chunks = list(receive_transcription(asr=asr, id=sender.id))
    assert TranscriptionChunk.join(chunks) == 'ONE ZERO ZERO ZERO ONE NINE O TWO ONE O CYRIL ONE AID CYRROW THREE'
    
    asr.stop_thread()