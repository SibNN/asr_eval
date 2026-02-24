# Class LongformCTC (defined in asr_eval/models/base/longform.py at lines 97-183)

class LongformCTC(asr_eval.models.base.interfaces.CTC):
    """A wrapper to apply a shortform CTC model to a longform audio.

    Longform transcriber means one being able to transcribe long audios.
    The current wrapper segments audio uniformly with overlaps, then
    averages the logprobs for all segments. By default averages with
    beta-distributed weights (:code:`averaging_weights='beta'`), because
    a model may be less certain on the edges of the segment.

    See also: :class:`~asr_eval.models.base.longform.LongformVAD`.
    """
    ...

    @typing.override
    def ctc_log_probs(self, waveforms: list[asr_eval.utils.types.FLOATS]) -> list[asr_eval.utils.types.FLOATS]:
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