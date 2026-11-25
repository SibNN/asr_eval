import sys
import types
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from asr_eval.bench.pipelines import get_pipeline
from asr_eval.models.base.interfaces import Transcriber
from asr_eval.models.qwen3_asr_wrapper import (
    ORUK_FOURIER_2B_MODEL_ID,
    ORUK_FOURIER_2B_REVISION,
    Qwen3ASRWrapper,
)


class _FakeTensor:
    def __init__(self, data: np.ndarray[Any, Any]):
        self._data = data
        self.shape = data.shape

    def __getitem__(self, item: Any) -> '_FakeTensor':
        return _FakeTensor(self._data[item])

    def tolist(self) -> Any:
        return self._data.tolist()


class _FakeBatchFeature(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.moved_to: tuple[Any, Any] | None = None

    def to(self, device: Any, dtype: Any = None) -> '_FakeBatchFeature':
        self.moved_to = (device, dtype)
        return self


def _install_fake_torch_and_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool = False,
    bf16_supported: bool = False,
    include_qwen3_cls: bool = True,
) -> dict[str, Any]:
    recorded: dict[str, Any] = {
        'processor_from_pretrained': [],
        'model_from_pretrained': [],
        'apply_calls': [],
        'generate_calls': [],
        'decode_calls': [],
        'last_batch': None,
    }

    fake_torch = types.ModuleType('torch')
    fake_torch.float32 = 'torch.float32'  # type: ignore[attr-defined]
    fake_torch.float16 = 'torch.float16'  # type: ignore[attr-defined]
    fake_torch.bfloat16 = 'torch.bfloat16'  # type: ignore[attr-defined]
    fake_torch.inference_mode = lambda: nullcontext()  # type: ignore[attr-defined]
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
    )
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> '_FakeProcessor':
            recorded['processor_from_pretrained'].append((model_name, kwargs))
            return cls()

        def apply_transcription_request(self, **kwargs: Any) -> _FakeBatchFeature:
            recorded['apply_calls'].append(kwargs)
            batch = _FakeBatchFeature(
                input_ids=_FakeTensor(np.array([[10, 11, 12]])),
                input_features=_FakeTensor(np.zeros((1, 8, 4), dtype=np.float32)),
            )
            recorded['last_batch'] = batch
            return batch

        def decode(self, tokens: Any, **kwargs: Any) -> list[str]:
            recorded['decode_calls'].append((tokens.tolist(), kwargs))
            return ['  hello from qwen3 asr  ']

    class _FakeModel:
        def __init__(self, dtype: Any):
            self.device: str = 'cpu'
            self.dtype: Any = dtype
            self.is_eval: bool = False

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> '_FakeModel':
            recorded['model_from_pretrained'].append((model_name, kwargs))
            return cls(dtype=kwargs.get('dtype'))

        def to(self, device: str) -> '_FakeModel':
            self.device = device
            return self

        def eval(self) -> '_FakeModel':
            self.is_eval = True
            return self

        def generate(self, **kwargs: Any) -> _FakeTensor:
            recorded['generate_calls'].append(kwargs)
            return _FakeTensor(np.array([[10, 11, 12, 101, 102]]))

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoProcessor = _FakeProcessor  # type: ignore[attr-defined]
    if include_qwen3_cls:
        fake_transformers.Qwen3ASRForConditionalGeneration = _FakeModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    return recorded


def test_default_model_loads_and_transcribes_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _install_fake_torch_and_transformers(monkeypatch, cuda_available=False)

    wrapper = Qwen3ASRWrapper()
    assert isinstance(wrapper, Transcriber)
    assert rec['processor_from_pretrained'] == [('Qwen/Qwen3-ASR-1.7B-hf', {})]
    assert rec['model_from_pretrained'] == [
        ('Qwen/Qwen3-ASR-1.7B-hf', {'dtype': 'torch.float32'})
    ]
    assert wrapper.model.device == 'cpu'
    assert wrapper.model.is_eval is True

    wav = np.zeros(1600, dtype=np.float32)
    text = wrapper.transcribe(wav)
    assert text == 'hello from qwen3 asr'
    assert 'language' not in rec['apply_calls'][0]
    assert rec['apply_calls'][0]['return_tensors'] == 'pt'
    assert rec['last_batch'].moved_to == ('cpu', 'torch.float32')
    assert rec['generate_calls'][0]['max_new_tokens'] == 512
    assert 'num_beams' not in rec['generate_calls'][0]
    assert rec['decode_calls'] == [([[101, 102]], {'return_format': 'transcription_only'})]


def test_oruk_fourier_2b_pins_revision_language_and_bf16_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _install_fake_torch_and_transformers(
        monkeypatch, cuda_available=True, bf16_supported=True
    )

    wrapper = Qwen3ASRWrapper(
        ORUK_FOURIER_2B_MODEL_ID,
        revision=ORUK_FOURIER_2B_REVISION,
        language='en',
        num_beams=3,
    )
    assert rec['processor_from_pretrained'] == [
        (ORUK_FOURIER_2B_MODEL_ID, {'revision': ORUK_FOURIER_2B_REVISION})
    ]
    assert rec['model_from_pretrained'] == [
        (
            ORUK_FOURIER_2B_MODEL_ID,
            {'dtype': 'torch.bfloat16', 'revision': ORUK_FOURIER_2B_REVISION},
        )
    ]
    assert wrapper.model.device == 'cuda'

    wav = np.ones(800, dtype=np.float32) * 0.1
    assert wrapper.transcribe(wav) == 'hello from qwen3 asr'
    assert rec['apply_calls'][0]['language'] == 'en'
    assert rec['generate_calls'][0]['num_beams'] == 3


def test_empty_waveform_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _install_fake_torch_and_transformers(monkeypatch)
    wrapper = Qwen3ASRWrapper()
    assert wrapper.transcribe(np.zeros(0, dtype=np.float32)) == ''
    assert rec['apply_calls'] == []
    assert rec['generate_calls'] == []


def test_old_transformers_raises_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_and_transformers(monkeypatch, include_qwen3_cls=False)
    with pytest.raises(ImportError, match='Qwen3ASRForConditionalGeneration'):
        Qwen3ASRWrapper()


def test_pipelines_are_registered() -> None:
    assert get_pipeline('qwen3-asr-1.7b-vad') is not None
    assert get_pipeline('oruk-fourier-2b-vad') is not None
