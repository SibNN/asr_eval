from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, override

import numpy as np

if TYPE_CHECKING:
    import speechbrain.inference.ASR

from asr_eval.streaming.buffer import ID_TYPE
from asr_eval.streaming.model import (
    OutputChunk, StreamingASR, Signal, TranscriptionChunk
)


__all__ = [
    'SpeechbrainStreaming',
]


class SpeechbrainStreaming(StreamingASR):
    """
    A speechbrain streaming model asr-streaming-conformer-gigaspeech.
    
    Adopted from Gradio example from here:
    https://huggingface.co/speechbrain/asr-streaming-conformer-librispeech
    
    Installation: see :doc:`/guide_installation` page.
    """
    
    def __init__(
        self,
        model_name: Literal[
            'speechbrain/asr-streaming-conformer-gigaspeech'
        ] = 'speechbrain/asr-streaming-conformer-gigaspeech',
        sampling_rate: int = 16_000,
    ):
        super().__init__(sampling_rate=sampling_rate)
        
        import speechbrain.inference.ASR # type: ignore
        from speechbrain.utils.dynamic_chunk_training import (
            DynChunkTrainConfig
        )
        
        self.config = DynChunkTrainConfig(24, 4)
        self.norm = True
        self.model = self.get_model(model_name=model_name)
        self.chunk_size = self.model.get_chunk_size_frames(self.config)
        self.contexts: dict[
            ID_TYPE, speechbrain.inference.ASR.ASRStreamingContext
        ] = defaultdict(
            lambda: self.model.make_streaming_context(self.config)
        )
    
    def get_model(
        self, model_name: str
    ) -> speechbrain.inference.ASR.StreamingASR:
        import speechbrain.inference.ASR
        
        model = speechbrain.inference.ASR.StreamingASR.from_hparams( # type: ignore
            source=model_name,
        )
        assert model is not None
        return model
    
    @override
    def _run(self):
        import torch
        
        while True:
            id, data, is_finished, end_time = (
                self.input_buffer.get_with_rechunking(self.chunk_size)
            )
            if data is not None:
                assert isinstance(data, np.ndarray)
                data_tensor = torch.tensor(
                    data.astype(np.float32),
                    dtype=torch.float32,
                    device='cuda',
                )
                if self.norm:
                    data_tensor /= (
                        max(1, torch.max(torch.abs(data_tensor)).item())
                    )
                with torch.no_grad():
                    text = self.model.transcribe_chunk(
                        self.contexts[id],
                        data_tensor.unsqueeze(0)
                    )[0]
                    
                # speechbrain starts a new word from space
                text = text.lstrip()
                
                if len(text):
                    self.output_buffer.put(OutputChunk(
                        data=TranscriptionChunk(text=text),
                        seconds_processed=end_time,
                    ), id=id)
            if is_finished:
                self.output_buffer.put(OutputChunk(
                    data=Signal.FINISH, seconds_processed=end_time
                ), id=id)
                self.contexts.pop(id, None)
    
    @property
    @override
    def audio_type(self) -> Literal['float']:
        return 'float'