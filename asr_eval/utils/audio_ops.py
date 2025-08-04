from pathlib import Path
import io
import tempfile
from typing import Iterator
from contextlib import contextmanager

import librosa
import soundfile as sf
import numpy as np

from .types import FLOATS


def speech_sample(repeats: int = 1) -> FLOATS:
    waveform = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000)[0] # type: ignore
    return np.concatenate([waveform] * repeats) # type: ignore


def waveform_to_bytes(waveform: FLOATS, sampling_rate: int = 16_000, format: str = 'wav') -> bytes:
    sf.write(buffer := io.BytesIO(), waveform, samplerate=sampling_rate, format=format) # type: ignore
    buffer.seek(0)
    return buffer.read()


@contextmanager
def waveform_as_file(waveform: FLOATS) -> Iterator[Path]:
    '''
    Turns an audio with sampling rate 16_000 into file
    with audio_as_file(waveform) as audio_path:
        recognize_speech(path=audio_path)
    '''
    with tempfile.NamedTemporaryFile('wb', suffix='.wav') as f:
        sf.write(f, waveform, samplerate=16_000, format='wav') # type: ignore
        yield Path(f.name)