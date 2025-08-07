from dataclasses import replace
import operator
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core.annotation import Annotation

from asr_eval.segments.segment import DiarizationSegment
from asr_eval.utils.types import FLOATS
from asr_eval.segments.segment import AudioSegment


__all__ = [
    'SpeakerDiarizationWrapper',
    'get_speaker_ratios',
]


class SpeakerDiarizationWrapper:
    '''
    To use this, you first need to accept conditions
    here https://huggingface.co/pyannote/segmentation-3.0
    and here https://huggingface.co/pyannote/speaker-diarization-3.0
    
    Then specify your HF token in the environmental variable. For VS Code, you can just create
    .env file in the project dir with one line: HF_TOKEN=<your_token>
    '''
    pipeline: SpeakerDiarization
    
    def __init__(
        self,
        segmentation_batch_size: int = 256,
        embedding_batch_size: int = 128,
    ):
        self.pipeline = SpeakerDiarization(
            segmentation='pyannote/segmentation-3.0',
            segmentation_batch_size=segmentation_batch_size,
            embedding='speechbrain/spkrec-ecapa-voxceleb',
            embedding_batch_size=embedding_batch_size,
            clustering='AgglomerativeClustering',
            embedding_exclude_overlap=True,
        )
        
        self.pipeline.instantiate({ # type: ignore
                'clustering': {
                    'method': 'centroid',
                    'min_cluster_size': 12,
                    'threshold': 0.7045654963945799,
                },
                'segmentation': {
                    'min_duration_off': 0.0,
                }
            }
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def __call__(
        self, mono_sound: FLOATS, sampling_rate: int = 16_000
    ) -> tuple[list[DiarizationSegment], FLOATS]:
        annotation: Annotation
        speaker_embeddings: FLOATS
        annotation, speaker_embeddings = self.pipeline.apply({ # type: ignore
            'waveform': torch.tensor(mono_sound, dtype=torch.float32).unsqueeze(0),
            'sample_rate': sampling_rate,
        }, return_embeddings=True)
        segments = [
            DiarizationSegment(
                start_time=float(turn.start),
                end_time=float(turn.end),
                speaker_idx=int(annotation.labels().index(speaker)),
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True) # type: ignore
        ]
        segments, speaker_embeddings = _reorder_speakers(segments, speaker_embeddings)
        return segments, speaker_embeddings


def _reorder_speakers(
    segments: list[DiarizationSegment],
    speaker_embeddings: FLOATS,
) -> tuple[list[DiarizationSegment], FLOATS]:
    """
    Assigns index 0 to the speaker with the longest total speech duration,
    then index 1, and so on.
    1) Returns same segments with new speaker ids (not inplace).
    2) Reorders 'speaker_embeddings' field accordingly.
    """
    if len(segments) == 0:
        return segments, speaker_embeddings

    segments_df = pd.DataFrame([
        {'start': s.start_time, 'end': s.end_time, 'speaker_idx': s.speaker_idx}
        for s in segments
    ])
    
    speaker_durations = (  # <-- old speaker ids here
        segments_df
        .set_index('speaker_idx') # type: ignore
        .assign(duration=lambda df: df.end - df.start) # type: ignore
        .groupby('speaker_idx')['duration']
        .sum()  # to series, total speech time for each speaker_idx
        .sort_index()
    )
    speakers_new_ids = ( # type: ignore
        speaker_durations # type: ignore
        .rank(method='first', ascending=False)
        .astype(int)
        .values  # to numpy array
        - 1  # .rank() enumerates from 1, we need to enumerate from 0
    )
    
    segments = [replace(s, speaker_idx=speakers_new_ids[s.speaker_idx]) for s in segments]
    speaker_embeddings = speaker_embeddings.copy()[np.argsort(speakers_new_ids)] # type: ignore

    return segments, speaker_embeddings


def get_speaker_ratios(
    segment: AudioSegment,
    diarization_segments: list[DiarizationSegment]
) -> dict[int, float]:
    '''
    Calculates overlaps of a given segment with all the diarization segments, and return a dict,
    where keys are speaker indices and values are percents of their presence in the segment. The
    dict is sorted starting from the highest percent. If there are no overlaps, returns an empty dict.
    '''
    for dseg1 in diarization_segments:
        for dseg2 in diarization_segments:
            if dseg1 is not dseg2 and dseg1.speaker_idx == dseg2.speaker_idx:
                assert dseg1.overlap_seconds(dseg2) == 0
        
    results: dict[int, float] = defaultdict(float)
    for diar_segment in diarization_segments:
        if (overlap := segment.overlap_seconds(diar_segment)) > 0:
            results[diar_segment.speaker_idx] += overlap / segment.duration
    
    return dict(sorted(results.items(), key=operator.itemgetter(1)))