from __future__ import annotations
from pathlib import Path
import io
import tempfile
from typing import TYPE_CHECKING, Iterator, Literal
from contextlib import contextmanager

import numpy as np

if TYPE_CHECKING:
    import torchaudio.transforms
    import pydub

from asr_eval.utils.types import FLOATS, INTS


__all__ = [
    'waveform_to_bytes',
    'merge_synthetic_speech',
    'waveform_as_file',
    'convert_audio_format',
]


def waveform_to_bytes_buffer(
    waveform: FLOATS, sampling_rate: int = 16_000, format: str = 'wav'
) -> io.BytesIO:
    """Converts a waveform into :code:`io.BytesIO` buffer of WAV bytes
    (or another format passed as :code:`format` argument).
    
    Uses soundfile library.
    """
    
    import soundfile as sf

    sf.write( # type: ignore
        buffer := io.BytesIO(),
        waveform,
        samplerate=sampling_rate,
        format=format,
    )
    buffer.seek(0)
    return buffer


def waveform_to_bytes(
    waveform: FLOATS, sampling_rate: int = 16_000, format: str = 'wav'
) -> bytes:
    """Converts a waveform into WAV bytes (or another format passed as
    :code:`format` argument).
    """
    
    return waveform_to_bytes_buffer(
        waveform,
        sampling_rate=sampling_rate,
        format=format,
    ).read()


def waveform_to_pydub(
    waveform: FLOATS, sampling_rate: int = 16_000
) -> pydub.AudioSegment:
    """Converts a waveform into :code:`pydub.AudioSegment`.
    
    Requires pydub to be installed. This function is currently unused
    and pydub is not in requirements.
    """
    
    import pydub
    buffer = waveform_to_bytes_buffer(waveform, sampling_rate=sampling_rate)
    return pydub.AudioSegment.from_file(buffer) # type: ignore


_TORCH_RESAMPLE_KERNELS: dict[
    tuple[int, int], torchaudio.transforms.Resample
] = {}


def resample(
    waveform: FLOATS,
    from_sampling_rate: int = 16_000,
    to_sampling_rate: int = 16_000,
) -> FLOATS:
    """Resamples the audio using torchaudio (preferably) or
    librosa if torchaudio is not available.
    
    Example:
        >>> import librosa
        >>> waveform, _ = librosa.load('tests/testdata/formula1.mp3', sr=22050)
        >>> from asr_eval.utils.audio_ops import resample
        >>> audio = resample(waveform, 22050, 16000)
    """
    
    global _TORCH_RESAMPLE_KERNELS
    if to_sampling_rate != from_sampling_rate:
        try:
            import torch
            from torchaudio.transforms import Resample
            
            key = (from_sampling_rate, to_sampling_rate)
            if not key in _TORCH_RESAMPLE_KERNELS:
                _TORCH_RESAMPLE_KERNELS[key] = Resample(
                    orig_freq=from_sampling_rate,
                    new_freq=to_sampling_rate,
                )
            op = _TORCH_RESAMPLE_KERNELS[key]
            waveform = (
                op(torch.tensor(waveform, dtype=torch.float32))
                .numpy() # type: ignore
            )
        except ImportError:
            try:
                import librosa
            except ImportError as e:
                raise RuntimeError(
                    'Both torchaudio and librosa are not installed'
                    ', cannot do audio resampling.'
                ) from e
            waveform = librosa.resample( # type: ignore
                waveform,
                orig_sr=from_sampling_rate,
                target_sr=to_sampling_rate,
            )
    return waveform


def convert_audio_format(
    waveform: FLOATS,
    to_audio_type: Literal['float', 'int', 'bytes', 'wav'] = 'float',
) -> FLOATS | INTS | bytes:
    """
    Converts a waveform with sampling rate 16000 into one of the
    pre-defined formats:
    
    - 'float': float values, preferrably from -1 to 1. Does nothing
        because this is the same as input format.
    - 'int': :code:`np.int16` values.
    - 'bytes': 2 bytes per frame.
    - 'wav': 2 bytes per frame plus WAV header.
    
    TODO find some python library that already supports these formats
    and conversions, or design this better.
    """
    
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
    """Merges speech segments using silent pauses of random length in
    :code:`pause_range`.
    
    Is suitable to construct a longform synthetic speech.
    """
    
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
    """Turns a waveform into a file. The file is deleted on exit from
    the context.
    
    Example:
        >>> with waveform_as_file(waveform) as audio_path:  # doctest: +SKIP
        ...     recognize_speech(path=audio_path)
    """
    
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile('wb', suffix='.wav') as f:
        sf.write(f, waveform, samplerate=16_000, format='wav') # type: ignore
        yield Path(f.name)