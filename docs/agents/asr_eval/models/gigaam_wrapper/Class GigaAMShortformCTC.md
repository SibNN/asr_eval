# Class GigaAMShortformCTC (defined in asr_eval/models/gigaam_wrapper.py at lines 136-264)

class GigaAMShortformCTC(asr_eval.models.gigaam_wrapper.GigaAMShortformBase, asr_eval.models.base.interfaces.CTC):
    """A GigaAM CTC model. Supports different versions (see
    :code:`version` parameter): "v2", "v3", "v3_e2e".

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
        # we have two base classes: GigaAMShortformBase and CTC
        # usually we want to use GigaAMShortformBase base class to call
        # .transcribe()
        ...

    @property
    @typing.override
    def blank_id(self) -> int:
    ...

    @property
    @typing.override
    def tick_size(self) -> float:
    ...

    @property
    @typing.override
    def vocab(self) -> tuple[str, ...]:
    ...

    @typing.override
    def ctc_log_probs(self, waveforms: list[asr_eval.utils.types.FLOATS]) -> list[asr_eval.utils.types.FLOATS]:
    ...