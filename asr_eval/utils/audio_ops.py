from pathlib import Path
import io
import tempfile
from typing import Iterator
from contextlib import contextmanager

import pydub
import librosa
import soundfile as sf
import numpy as np

from .types import FLOATS


__all__ = [
    'speech_sample',
    'waveform_to_bytes',
    'waveform_to_pydub',
    'merge_synthetic_speech',
    'waveform_as_file',
]


def speech_sample(repeats: int = 1) -> FLOATS:
    '''
    A sample waveform with Russian speech.
    '''
    waveform = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000)[0] # type: ignore
    return np.concatenate([waveform] * repeats) # type: ignore


def waveform_to_bytes(waveform: FLOATS, sampling_rate: int = 16_000, format: str = 'wav') -> bytes:
    '''
    Converts a waveform into bytes.
    '''
    sf.write(buffer := io.BytesIO(), waveform, samplerate=sampling_rate, format=format) # type: ignore
    buffer.seek(0)
    return buffer.read()


def waveform_to_pydub(waveform: FLOATS, sampling_rate: int = 16_000) -> pydub.AudioSegment:
    '''
    Converts a waveform into pydub.AudioSegment.
    '''
    bytes = waveform_to_bytes(waveform)
    buffer = io.BytesIO(bytes)
    return pydub.AudioSegment.from_file(buffer) # type: ignore


def merge_synthetic_speech(
    waveforms: list[FLOATS],
    sampling_rate: int = 16_000,
    pause_range: tuple[float, float] = (0.2, 1.2),
    random_seed: int | None = None,
) -> FLOATS:
    '''
    Merges synthetic speech segments with silence pauses of random lengths.
    '''
    segments: list[FLOATS] = []
    rng = np.random.default_rng(random_seed)
    for i, waveform in enumerate(waveforms):
        segments.append(waveform)
        if i != len(waveforms) - 1:
            pause_size = int(rng.uniform(*pause_range) * sampling_rate)
            segments.append(np.zeros(pause_size))
    
    return np.concatenate(segments)

@contextmanager
def waveform_as_file(waveform: FLOATS) -> Iterator[Path]:
    '''
    Turns an audio with sampling rate 16_000 into file that is deleted afterwards.
    
    Example:
    with audio_as_file(waveform) as audio_path:
        recognize_speech(path=audio_path)
    '''
    with tempfile.NamedTemporaryFile('wb', suffix='.wav') as f:
        sf.write(f, waveform, samplerate=16_000, format='wav') # type: ignore
        yield Path(f.name)