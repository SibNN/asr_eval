from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

import soundfile as sf

from ..utils.types import FLOATS


@contextmanager
def audio_as_file(waveform: FLOATS) -> Iterator[Path]:
    '''
    Turns an audio with sampling rate 16_000 into file, TODO check if it works
    with audio_as_file(waveform) as temp_file_path:
        recognize_speech(path=temp_file_path)
    '''
    with tempfile.NamedTemporaryFile('wb') as f:
        sf.write(f, waveform, samplerate=16_000, format='wav') # type: ignore
        yield Path(f.name)
    f.unlink()