from typing import override
import warnings
import numpy as np
import torch
from gigaam.vad_utils import segment_audio as _segment_audio

from asr_eval.models.base.interfaces import Segmenter

from ..segments.segment import AudioSegment
from ..utils.types import FLOATS


class PyannoteSegmenter(Segmenter):
    def __init__(self, max_duration: float = 22, min_duration: float = 15):
        self.max_duration = float(max_duration)
        self.min_duration = float(min_duration)
    
    @override
    def __call__(self, waveform: FLOATS, initial_threshold: float = 0.2) -> list[AudioSegment]:
        '''
        VAD-based longform audio segmenter based on Pyannote wrapper from GigaAM package
        
        No model loading in __init__ because _segment_audio uses global model object
        and loads it on the first call.
        
        The params are taken from transcribe_longform method from gigaam package. 
        
        Problem with gigaam.vad_utils.segment_audio is that it can return weird output,
        for example, (0.0, 0.0), (0.03096875, 39.509) - the first chunk has zero length,
        and the second is longer than max_duration.
        
        This wrapper addresses this by 1) removing chunks shorter than 0.1 sec, and
        2) performs a uniform split of chunks longer than max_duration
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, boundaries = _segment_audio(
            torch.tensor(waveform * 32768, dtype=torch.int16).clone(),
            16_000,
            max_duration=self.max_duration,
            min_duration=self.min_duration,
            new_chunk_threshold=0.2,
            device=device,
        )
        
        segments: list[AudioSegment] = []
        for segment_start, segment_end in boundaries:
            segment = AudioSegment(segment_start, segment_end)
            if segment.duration < 0.1:
                continue
            elif segment.duration > self.max_duration:
                warnings.warn(
                    f'Warning! Segment is too large ({segment.duration:.1f} sec)'
                    ', splitting uniformly',
                )
                for delta in np.arange(0, segment.duration, self.max_duration):
                    subsegment_start = segment.start_time + float(delta)
                    subsegment_end = min(subsegment_start + self.max_duration, segment.end_time)
                    segments.append(AudioSegment(subsegment_start, subsegment_end))
            else:
                segments.append(segment)
        
        return segments