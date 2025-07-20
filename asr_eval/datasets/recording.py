from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from gigaam.model import GigaAMASR

from ..align.data import Token, MultiVariant
from ..align.parsing import parse_multivariant_string
from ..align.timings import fill_word_timings_inplace


@dataclass
class Recording:
    """
    An audio sample to test ASR systems. May refer to a huggingface dataset sample.
    
    This is useful because HF dataset samples themselves cannot keep transcriptions in form of
    list[Token | MultiVariant], and also for tracking source samples when saving predictions
    (such as RecordingStreamingEvaluation with .recording field).
    
    """
    transcription: str
    transcription_words: list[Token | MultiVariant]
    waveform: npt.NDArray[np.floating[Any]] | None = None
    
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
        sample: dict[str, Any],
        name: str | None = None,
        split: str | None = None,
        index: int | None = None,
        use_gigaam: GigaAMASR | None = None,
    ) -> Recording:
        assert sample['audio']['sampling_rate'] == 16_000
        waveform = sample['audio']['array']
        text = sample['transcription']
        
        # this will also work for texts without multivariant blocks
        transcription_words = parse_multivariant_string(text)
        
        if use_gigaam is not None:
            fill_word_timings_inplace(use_gigaam, waveform, transcription_words)
            # transcription_words = cast(
            #     list[Token | MultiVariant], get_word_timings_simple(fill_timings, waveform, text)
            # )
        
        return Recording(
            waveform=waveform,
            transcription=text,
            transcription_words=transcription_words,
            hf_dataset_name=name,
            hf_dataset_split=split,
            hf_dataset_index=index,
        )