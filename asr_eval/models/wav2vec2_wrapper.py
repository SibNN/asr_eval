from typing import override

from asr_eval.models.base.interfaces import CTC
from asr_eval.utils.types import FLOATS


__all__ = [
    'Wav2vec2Wrapper',
]


class Wav2vec2Wrapper(CTC):
    """A wrapper for wav2vec2 Hugging Face models.
    
    Requires :code:`transformers` package.

    Note:
        This does not support :code:`Wav2Vec2ProcessorWithLM`. This
        wrapper is in :class:`~asr_eval.models.base.interfaces.CTC`
        format: it returns log probs only. If you need LM, you may use
        :class:`~asr_eval.ctc.lm.CTCDecoderWithLM`.
    """
    def __init__(
        self,
        model_name: str = 'facebook/wav2vec2-base-960h',
    ):
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) # type: ignore
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device) # type: ignore

        # calculate the vocabulary in the CTC.vocab() format
        vocab_dict: dict[int, str] = {
            i: s for s, i in self.processor.tokenizer.vocab.items() # type: ignore
        }
        vocab_str = [
            vocab_dict[i] for i in range(self.processor.tokenizer.vocab_size) # type: ignore
        ]
        # replace pad token (= blank token) with '',
        # as required by the CTC.vocab() format
        vocab_str[self.processor.tokenizer.pad_token_type_id] = '' # type: ignore
        # replace | with space
        if '|' in vocab_str:
            vocab_str[vocab_str.index('|')] = ' '
        self._vocab = tuple(vocab_str)
    
    @override
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        import torch
        
        processed = self.processor( # type: ignore
            waveforms,
            return_tensors='pt', # type: ignore
            padding='longest', # type: ignore
            sampling_rate=16_000, # type: ignore
            return_attention_mask=True, # type: ignore
        ).to(self.device)
        input_lengths = self.model._get_feat_extract_output_lengths( # type: ignore
            processed.attention_mask.sum(-1) # type: ignore
        ).cpu().numpy() # type: ignore
        with torch.inference_mode():
            outputs = self.model(**processed)
            log_probs = torch.nn.functional.log_softmax( # type: ignore
                outputs.logits, dim=-1
            ).cpu().numpy()
        return [
            log_probs[i, :length] for i, length in enumerate(input_lengths) # type: ignore
        ]
    
    @property
    @override
    def blank_id(self) -> int:
        # <pad> is used as a blank token and for padding
        return self.processor.tokenizer.pad_token_type_id # type: ignore
    
    @property
    @override
    def tick_size(self) -> float:
        return (
            self.model.config.inputs_to_logits_ratio # type: ignore
            / self.processor.feature_extractor.sampling_rate # type: ignore
        )
    
    @property
    @override
    def vocab(self) -> tuple[str, ...]:
        return self._vocab