# Class VikhrBorealisWrapper (defined in asr_eval/models/vikhr_wrapper.py at lines 18-65)

class VikhrBorealisWrapper(asr_eval.models.base.interfaces.Transcriber):
    """A Vikhr Borealis wrapper.

    Loading a model takes a long time, around 2 min.

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    model: GenerationMixin

    extractor: WhisperFeatureExtractor

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...