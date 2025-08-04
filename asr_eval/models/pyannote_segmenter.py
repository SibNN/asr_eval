from typing import override
import torch
from gigaam.vad_utils import segment_audio as _segment_audio

from asr_eval.models.base.interfaces import Segmenter

from ..segments.segment import AudioSegment
from ..utils.types import FLOATS


class PyannoteSegmenter(Segmenter):
    @override
    def __call__(self, waveform: FLOATS) -> list[AudioSegment]:
        '''
        VAD-based longform audio segmenter based on Pyannote wrapper from GigaAM package
        
        No model loading in __init__ because _segment_audio uses global model object
        and loads it on the first call.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, boundaries = _segment_audio(
            torch.tensor(waveform * 32768, dtype=torch.int16).clone(),
            16_000,
            max_duration=22.,
            min_duration=15.,
            new_chunk_threshold=0.2,
            device=device,
        )
        return [
            AudioSegment(segment_start, segment_end)
            for segment_start, segment_end in boundaries
            if segment_end - segment_start > 0.1
        ]