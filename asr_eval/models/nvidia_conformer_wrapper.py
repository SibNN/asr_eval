from pathlib import Path

import nemo.collections.asr as nemo_asr


class NvidiaConformerWrapper:
    '''
    A Russian non-streaming model
    https://huggingface.co/nvidia/stt_ru_conformer_ctc_large
    TODO
    '''
    def __init__(self):
        self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained( # type: ignore
            model_name="stt_ru_conformer_ctc_large"
        )
    
    def transcribe(self, audio_paths: list[Path]) -> list[str]:
        outputs = asr_model.transcribe([str(path) for path in audio_paths]) # type: ignore
        return [output.text for output in outputs] # type: ignore