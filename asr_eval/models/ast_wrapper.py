from typing import Literal

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from asr_eval.segments.chunking import chunk_audio
from asr_eval.utils.types import FLOATS


class AudioSpectrogramTransformer:
    def __init__(
        self,
        model_path: str = 'MIT/ast-finetuned-audioset-10-10-0.4593',
        device: str = 'cuda',
    ):
        self.device = device
        self.feature_extractor: ASTFeatureExtractor = ASTFeatureExtractor.from_pretrained(model_path) # type: ignore
        self.model: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(model_path).to(device) # type: ignore
        self.labels: list[str] = [
            self.model.config.id2label[id] # type: ignore
            for id in range(self.model.config.num_labels) # type: ignore
        ]
        self.min_frames = 240  # min waveform length to pass into feature_extractor and model
    
    @torch.inference_mode()
    def predict_on_batch(self, waveforms: FLOATS, sampling_rate: int = 16_000) -> FLOATS:
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors='pt'
        )
        inputs = inputs.to(self.device) # type: ignore
        return self.model(**inputs).logits.detach().cpu().numpy()
    
    def predict_longform(
        self,
        waveform: FLOATS,
        batch_size: int = 32,
        segment_length: float = 10,  # a train-time value for AST
        segment_shift: float = 5,
        sampling_rate: int = 16_000,
        min_length: float = 1,  # if less, don't want to predict, too short and considered OOD
    ) -> FLOATS:
        segments = chunk_audio(
            len(waveform) / sampling_rate,
            segment_length=segment_length,
            segment_shift=segment_shift,
            last_chunk_mode='same_length',
        )
        waveforms = [waveform[segment.slice()] for segment in segments]
        batches = [
            np.stack(waveforms[i:i + batch_size], axis=0)
            for i in range(0, len(waveforms), batch_size)
        ]
        
        if (
            segments[0].duration < min_length
            or len(waveforms[0]) < self.min_frames
        ):
            return np.zeros((0, len(self.labels)))
        
        logits = np.concatenate([
            self.predict_on_batch(batch, sampling_rate=sampling_rate) for batch in batches
        ], axis=0)
        
        return logits

    def plot_top_classes(self, logits: FLOATS, top_by: Literal['max', 'mean'] = 'max'):
        # logits have shape (n_segments, n_classes)
        reduction = {'max': np.max, 'mean': np.mean}[top_by]
        classes_to_display: list[int] = np.argsort(reduction(logits, axis=0))[:-10:-1].tolist()
        for cls_idx in classes_to_display:
            plt.plot(logits[:, cls_idx], label=self.labels[cls_idx]) # type: ignore
        plt.legend() # type: ignore
        plt.show() # type: ignore