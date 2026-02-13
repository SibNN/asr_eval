from __future__ import annotations
from typing import TYPE_CHECKING, override, cast

import numpy as np

if TYPE_CHECKING:
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from nemo.collections.asr.parts.mixins.transcription import (
        TranscriptionMixin
    )

from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS


__all__ = [
    'NvidiaNemoWrapper',
]


class NvidiaNemoWrapper(Transcriber):
    """
    A Nvidia NEMO wrapper
    
    Installation: see :doc:`/guide_installation` page.

    Some of the available models (many more are available):
    1. "nvidia/canary-1b-v2"
        - NOTE: Specify language, example:
          :code:`inference_kwargs={'source_lang': 'ru', 'target_lang': 'ru'}`
        - NOTE: in Nemo :code:`beam_size=1` by default
    2. "nvidia/parakeet-tdt-0.6b-v3"
        - NOTE: Supports torch.float16 or torch.bfloat16 only with
          :code:`amp=True`
        - NOTE: in Nemo :code:`beam_size=2` by default
    3. "nvidia/stt_ru_fastconformer_hybrid_large_pc"
        NOTE: in Nemo :code:`beam_size=2` by default

    Dtypes:
        - for :code:`amp=True`, available dtypes are torch.float16,
          torch.bfloat16
        - for :code:`amp=False`, available dtypes are torch.float16,
          torch.bfloat16, torch.float32

    Notes:

    This wrapper is build using the following docs and examples:
    https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/transcribe_speech.py
    https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr
        /api.html#nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin
    
    The NEMO wrapper seems not to perform internal VAD; it raises OOM on
    too long audios. From the EncDecMultiTaskModel docstrings:
    "recommended length per file is between 5 and 25 seconds, but it is
    possible to pass a few hours long file if enough GPU memory is
    available".

    The :code:`.transcribe()` method of the NEMO's
    :code:`TranscriptionMixin` allows to pass :code:`timestamps=True`.
    It raises error for Canary, but returns timestamps for Parakeet and
    FastComformer. However, the output timestamps require postprocessing
    that is not implemented currently.

    Some of the models should support CTC interface and/or LM
    interation, but this is not implemented in asr_eval currently.
    
    To get the full list of available models, run:

    .. code-block:: python
    
        from nemo.collections.asr.models import ASRModel
        print(ASRModel.list_available_models())
    """
    
    model: TranscriptionMixin

    def __init__(
        self,
        model_name: str,
        inference_kwargs: dict[str, str] | None = None,
        verbose: bool = False,
        dtype: torch.dtype | str = 'float32',
        amp: bool = False,  # mixed precision inference
        beam_size: int | None = None,
    ):
        import torch
        from nemo.collections.asr.models import ASRModel
        
        self.inference_kwargs = inference_kwargs
        self.verbose = verbose
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.amp = amp

        if not verbose:
            from nemo.utils import logging
            from nemo.utils.nemo_logging import Logger
            logging.setLevel(Logger.ERROR) # type: ignore

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ASRModel.from_pretrained( # type: ignore
            model_name=model_name,
            map_location=torch.device(self.device),
        )

        if not amp:
            self.model = self.model.to(self.dtype) # type: ignore
        
        if beam_size is not None:
            decode_cfg = self.model.cfg.decoding # type: ignore
            decode_cfg.beam.beam_size = beam_size # type: ignore
            self.model.change_decoding_strategy(decode_cfg) # type: ignore
                
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        import torch
        
        with torch.amp.autocast(
            self.device, dtype=self.dtype, enabled=self.amp
        ):
            with torch.no_grad():
                hypothesis: Hypothesis = self.model.transcribe( # type: ignore
                    waveform.astype(np.float32),
                    timestamps=False,
                    verbose=self.verbose,
                    **(self.inference_kwargs or {}) # type: ignore
                )[0]
                
        if isinstance(hypothesis, list):
            text = ' '.join(hypothesis) # type: ignore  # is a list of strings
            # TODO review this issue
            text = (
                text
                .replace('<|endoftext|>.', '')
                .replace('<|endoftext|>', '')
            )
        elif isinstance(hypothesis, str):
            text = hypothesis
            text = (
                hypothesis
                .replace('<|endoftext|>.', '')
                .replace('<|endoftext|>', '')
            )
        else:
            text = cast(str, hypothesis.text) # type: ignore
            assert text is not None
        
        return text