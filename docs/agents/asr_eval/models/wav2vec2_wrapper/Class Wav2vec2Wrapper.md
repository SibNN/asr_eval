# Class Wav2vec2Wrapper (defined in asr_eval/models/wav2vec2_wrapper.py at lines 12-90)

class Wav2vec2Wrapper(asr_eval.models.base.interfaces.CTC):
    """A wrapper for wav2vec2 Hugging Face models.

    Requires :code:`transformers` package.

    Note:
        This does not support :code:`Wav2Vec2ProcessorWithLM`. This
        wrapper is in :class:`~asr_eval.models.base.interfaces.CTC`
        format: it returns log probs only. If you need LM, you may use
        :class:`~asr_eval.ctc.lm.CTCDecoderWithLM`.
    """
    ...

    @typing.override
    def ctc_log_probs(self, waveforms: list[asr_eval.utils.types.FLOATS]) -> list[asr_eval.utils.types.FLOATS]:
    ...

    @property
    @typing.override
    def blank_id(self) -> int:
        # <pad> is used as a blank token and for padding
        ...

    @property
    @typing.override
    def tick_size(self) -> float:
    ...

    @property
    @typing.override
    def vocab(self) -> tuple[str, ...]:
    ...