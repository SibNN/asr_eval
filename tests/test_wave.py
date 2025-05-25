import wave

import pytest
import librosa
import numpy as np
import numpy.typing as npt

"""
Check that both methods give the same waveform:
1) reading floats with librosa
1) reading wav bytes (to send into Vosk, for example)
"""

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_wave():
    # vosk.wav from https://github.com/alphacep/vosk-api/raw/eabd80a848de53e87e5943937146025d42ae570d/python/example/test.wav
    # adopted from https://github.com/alphacep/vosk-api/blob/eabd80a848de53e87e5943937146025d42ae570d/python/example/test_ep.py
    audio_path = 'tests/testdata/vosk.wav'
    
    # Load with librosa
    waveform_librosa: npt.NDArray[np.floating]
    waveform_librosa, rate_librosa = librosa.load(audio_path, sr=None) # type: ignore
    assert rate_librosa == 16_000

    # Load with wave
    wf = wave.open(audio_path, 'rb')
    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getcomptype() == 'NONE'
    assert wf.getframerate() == 16_000
    data = wf.readframes(wf.getnframes())
    waveform_from_bytes = np.frombuffer(data, dtype=np.int16)

    # compare
    assert all(np.int16(waveform_librosa * 32768) == waveform_from_bytes)
    
    # inverse convertion
    bytes_from_waveform = np.int16(waveform_librosa * 32768).tobytes()
    assert bytes_from_waveform == data