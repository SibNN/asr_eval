from typing import TYPE_CHECKING, override, cast

import numpy as np

if TYPE_CHECKING:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from .base.interfaces import Transcriber
from ..utils.types import FLOATS


__all__ = [
    'NvidiaCanaryWrapper',
]


class NvidiaCanaryWrapper(Transcriber):
    '''
    A Nvidia Canary wrapper
    
    TODO langs
    TODO timestamps=True will work?
    from EncDecMultiTaskModel docstrings:
    "recommended length per file is between 5 and 25 seconds"
    "but it is possible to pass a few hours long file if enough GPU memory is available"
    '''
    def __init__(self, model_name: str = 'nvidia/canary-1b-v2'):
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
        
        from nemo.utils import logging
        from nemo.utils.nemo_logging import Logger
        logging.setLevel(Logger.ERROR) # type: ignore

        self.model = cast(
            EncDecMultiTaskModel,
            ASRModel.from_pretrained(model_name=model_name), # type: ignore
        )
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        hypotheses: list[Hypothesis]
        hypotheses = self.model.transcribe( # type: ignore
            waveform.astype(np.float32),
            source_lang='ru',
            target_lang='ru',
            timestamps=False,
            verbose=False,
        )
        text = hypotheses[0].text
        assert text is not None
        return text