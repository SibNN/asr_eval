from typing import Any, cast, override

from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS

__all__ = [
    'ORUK_FOURIER_2B_MODEL_ID',
    'ORUK_FOURIER_2B_REVISION',
    'Qwen3ASRWrapper',
]

ORUK_FOURIER_2B_MODEL_ID = 'oruk/oruk-fourier-2b'
ORUK_FOURIER_2B_REVISION = '8c2a8871f1304aa2f634da4fb95e24438c4945b9'


class Qwen3ASRWrapper(Transcriber):
    """A wrapper for native Hugging Face Transformers Qwen3-ASR models.

    Supports ``Qwen3ASRForConditionalGeneration`` checkpoints such as
    ``Qwen/Qwen3-ASR-1.7B-hf`` and ``oruk/oruk-fourier-2b`` (English-only,
    CC BY-NC 4.0) using native ``transformers`` without ``trust_remote_code``.

    Example:
        >>> model = Qwen3ASRWrapper('Qwen/Qwen3-ASR-1.7B-hf')  # doctest: +SKIP
        >>> fourier = Qwen3ASRWrapper(  # doctest: +SKIP
        ...     ORUK_FOURIER_2B_MODEL_ID,
        ...     revision=ORUK_FOURIER_2B_REVISION,
        ...     language='en',
        ... )
        >>> text = fourier.transcribe(waveform)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen3-ASR-1.7B-hf',
        *,
        revision: str | None = None,
        language: str | None = None,
        device: str | None = None,
        dtype: str = 'auto',
        max_new_tokens: int = 512,
        num_beams: int | None = None,
    ):
        import torch
        import transformers

        model_cls: Any = getattr(transformers, 'Qwen3ASRForConditionalGeneration', None)
        processor_cls: Any = getattr(transformers, 'AutoProcessor', None)
        if model_cls is None or processor_cls is None:
            raise ImportError(
                'Qwen3ASRWrapper requires a Hugging Face transformers release '
                'with Qwen3ASRForConditionalGeneration support '
                '(upgrade with `pip install -U transformers`).'
            )

        self.model_name = model_name
        self.revision = revision
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype == 'auto':
            use_bf16 = (
                self.device.startswith('cuda')
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            resolved_dtype: Any = torch.bfloat16 if use_bf16 else torch.float32
        else:
            resolved_dtype = getattr(torch, dtype)

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs['revision'] = revision

        self._torch: Any = torch
        self.processor: Any = processor_cls.from_pretrained(model_name, **load_kwargs)
        self.model: Any = model_cls.from_pretrained(
            model_name,
            dtype=resolved_dtype,
            **load_kwargs,
        ).to(self.device).eval()

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        if len(waveform) == 0:
            return ''

        request_kwargs: dict[str, Any] = {
            'audio': waveform,
            'return_tensors': 'pt',
        }
        if self.language is not None:
            request_kwargs['language'] = self.language

        inputs: Any = self.processor.apply_transcription_request(**request_kwargs)
        inputs = inputs.to(self.model.device, self.model.dtype)

        gen_kwargs: dict[str, Any] = {'max_new_tokens': self.max_new_tokens}
        if self.num_beams is not None:
            gen_kwargs['num_beams'] = self.num_beams

        with self._torch.inference_mode():
            output_ids: Any = self.model.generate(**inputs, **gen_kwargs)

        input_len = int(inputs['input_ids'].shape[1])
        new_tokens: Any = output_ids[:, input_len:]
        decoded = cast(
            str | list[str],
            self.processor.decode(new_tokens, return_format='transcription_only'),
        )
        text = decoded if isinstance(decoded, str) else (decoded[0] if decoded else '')
        return text.strip()
