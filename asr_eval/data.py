from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gigaam.model import GigaAMASR

from asr_eval.align.data import Token, MultiVariant
from asr_eval.align.parsing import parse_multivariant_string
from asr_eval.streaming.timings import get_word_timings

if TYPE_CHECKING:
    from asr_eval.streaming.evaluation import RecordingStreamingEvaluation


@dataclass
class Recording:
    """
    An audio sample to test ASR systems. May refer to a huggingface dataset sample.
    """
    transcription: str
    transcription_words: list[Token | MultiVariant]
    waveform: npt.NDArray[np.floating] | None = None
    
    hf_dataset_name: str | None = None
    hf_dataset_split: str | None = None
    hf_dataset_index: int | None = None
    
    evals: RecordingStreamingEvaluation | None = None
    
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
        
        if use_gigaam is not None:
            transcription_words = cast(
                list[Token | MultiVariant], get_word_timings(use_gigaam, waveform, text)
            )
        else:
            # this will also work for texts without multivariant blocks
            transcription_words = parse_multivariant_string(text)
        
        return Recording(
            waveform=waveform,
            transcription=text,
            transcription_words=transcription_words,
            hf_dataset_name=name,
            hf_dataset_split=split,
            hf_dataset_index=index,
        )