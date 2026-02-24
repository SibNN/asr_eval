# Class VoskV54 (defined in asr_eval/models/vosk54_wrapper.py at lines 20-146)

class VoskV54(asr_eval.models.base.interfaces.Transcriber):
    """A wrapper for Vosk 0.54 model.

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...

    def batch_transcribe(self, waveforms: list[asr_eval.utils.types.FLOATS]) -> list[str]:
    ...