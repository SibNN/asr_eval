from pathlib import Path
import io
import tempfile
from typing import Iterator, Literal
from contextlib import contextmanager

import pydub
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio

from .types import FLOATS, INTS


__all__ = [
    'speech_sample',
    'waveform_to_bytes',
    'waveform_to_pydub',
    'merge_synthetic_speech',
    'waveform_as_file',
    'convert_audio_format',
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


def resample(
    waveform: FLOATS,
    from_sampling_rate: int = 16_000,
    to_sampling_rate: int = 16_000,
) -> FLOATS:
    '''
    Resamples the audio.
    
    Note that if `to_sampling_rate != from_sampling_rate`, this function uses
    `torchaudio.functional.resample`. If `.prepare_audio_format()` is called multiple
    times, it is more efficient to use a precomputed `torchaudio.transforms.Resample`,
    see https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.resample.html
    '''
    if to_sampling_rate != from_sampling_rate:
        waveform = torchaudio.functional.resample(
            torch.tensor(waveform),
            orig_freq=from_sampling_rate,
            new_freq=to_sampling_rate,
        ).numpy() # type: ignore
    return waveform


def convert_audio_format(
    waveform: FLOATS,
    to_audio_type: Literal['float', 'int', 'bytes', 'wav'] = 'float',
) -> FLOATS | INTS | bytes:
    '''
    Accepts a float waveform with a given sampling rate. Converts to the required format
    
    'float': float values, preferrably from -1 to 1
    'int': np.int16 values
    'bytes': 2 bytes per frame
    'wav': 2 bytes per frame and WAV header for each audio chunk
    
    Example conversions between formats:
    'float' -> 'int': audio = (audio * 32768).astype(np.int16)
    'int' -> 'bytes': audio = audio.tobytes()
    'float' -> 'wav': audio = asr_eval.utils.audio_ops.waveform_to_bytes(audio)
    'bytes' -> 'int':  audio = np.frombuffer(audio, dtype=np.int16)
    
    TODO find some python library that already supports these formats and conversions
    or design this better
    '''
    match to_audio_type:
        case 'float':
            return waveform
        case 'int':
            return (waveform * 32768).astype(np.int16)
        case 'bytes':
            return (waveform * 32768).astype(np.int16).tobytes()
        case 'wav':
            return waveform_to_bytes(waveform)


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