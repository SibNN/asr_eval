from collections import defaultdict
from typing import Literal, override

import torch
import numpy as np
import speechbrain.inference.ASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

from ..streaming.buffer import ID_TYPE
from ..streaming.model import OutputChunk, StreamingASR, Signal, TranscriptionChunk


__all__ = [
    'SpeechbrainStreaming',
]


class SpeechbrainStreaming(StreamingASR):
    """
    A asr-streaming-conformer-gigaspeech model.
    
    Adopted from Gradio example from here:
    https://huggingface.co/speechbrain/asr-streaming-conformer-librispeech
    """
    def __init__(
        self,
        sampling_rate: int = 16_000,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.config = DynChunkTrainConfig(24, 4)
        self.norm = True
        self.model = self.get_model()
        self.chunk_size = self.model.get_chunk_size_frames(self.config)
        self.contexts: dict[ID_TYPE, speechbrain.inference.ASR.ASRStreamingContext] = defaultdict(
            lambda: self.model.make_streaming_context(self.config)
        )
    
    def get_model(self) -> speechbrain.inference.ASR.StreamingASR:
        model = speechbrain.inference.ASR.StreamingASR.from_hparams( # type: ignore
            source="speechbrain/asr-streaming-conformer-gigaspeech",
        )
        assert model is not None
        return model
    
    @override
    def _run(self):
        while True:
            id, data, is_finished, end_time = self.input_buffer.get_with_rechunking(self.chunk_size)
            if data is not None:
                assert isinstance(data, np.ndarray)
                data_tensor = torch.tensor(data.astype(np.float32), dtype=torch.float32, device='cuda')
                if self.norm:
                    data_tensor /= max(1, torch.max(torch.abs(data_tensor)).item())
                with torch.no_grad():
                    text = self.model.transcribe_chunk(self.contexts[id], data_tensor.unsqueeze(0))[0]
                text = text.lstrip()  # speechbrain starts a new word from space
                if len(text):
                    self.output_buffer.put(OutputChunk(
                        data=TranscriptionChunk(text=text),
                        seconds_processed=end_time,
                    ), id=id)
            if is_finished:
                self.output_buffer.put(OutputChunk(data=Signal.FINISH, seconds_processed=end_time), id=id)
                self.contexts.pop(id, None)
    
    @property
    @override
    def audio_type(self) -> Literal['float']:
        return 'float'