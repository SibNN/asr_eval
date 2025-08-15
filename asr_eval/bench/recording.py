from __future__ import annotations

from dataclasses import dataclass

from ..models.gigaam_wrapper import GigaAMShortformCTC
from .datasets import AudioSample
from ..utils.types import FLOATS
from ..align.transcription import MultiVariantTranscription, SingleVariantTranscription
from ..align.parsing import parse_multivariant_string
from ..align.timings import fill_word_timings_inplace


__all__ = [
    'Recording',
]


@dataclass
class Recording:
    """
    NOTE: A legacy class, TODO remove?
    
    An audio sample to test ASR systems. May refer to a huggingface dataset sample.
    
    This is useful because HF dataset samples themselves cannot keep transcriptions in form of
    list[Token | MultiVariant], and also for tracking source samples when saving predictions
    (such as RecordingStreamingEvaluation with .recording field).
    
    """
    transcription: MultiVariantTranscription | SingleVariantTranscription
    waveform: FLOATS | None = None
    
    hf_dataset_name: str | None = None
    hf_dataset_split: str | None = None
    hf_dataset_index: int | None = None
    
    # evals: RecordingStreamingEvaluation | None = None
    
    @property
    def hf_uid(self) -> str:
        return f'{self.hf_dataset_name}/{self.hf_dataset_split}/{self.hf_dataset_index}'
    
    @classmethod
    def from_sample(
        cls,
        sample: AudioSample,
        name: str | None = None,
        split: str | None = None,
        index: int | None = None,
        use_gigaam: GigaAMShortformCTC | None = None,
    ) -> Recording:
        assert sample['audio']['sampling_rate'] == 16_000
        waveform = sample['audio']['array']
        text = sample['transcription']
        
        # this will also work for texts without multivariant blocks
        transcription = parse_multivariant_string(text)
        
        if use_gigaam is not None:
            fill_word_timings_inplace(use_gigaam, waveform, transcription)
        
        return Recording(
            waveform=waveform,
            transcription=transcription,
            hf_dataset_name=name,
            hf_dataset_split=split,
            hf_dataset_index=index,
        )