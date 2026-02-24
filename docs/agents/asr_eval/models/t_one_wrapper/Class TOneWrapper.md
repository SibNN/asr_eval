# Class TOneWrapper (defined in asr_eval/models/t_one_wrapper.py at lines 80-106)

class TOneWrapper(asr_eval.models.base.interfaces.Transcriber):
    """A non-streaming wrapper for T-One model.

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...