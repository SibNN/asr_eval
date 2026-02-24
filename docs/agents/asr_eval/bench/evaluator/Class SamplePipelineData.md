# Class SamplePipelineData (defined in asr_eval/bench/evaluator.py at lines 235-261)

@dataclasses.dataclass
class SamplePipelineData:
    """A field of the
    :class:`~asr_eval.bench.evaluator.DatasetData` dataclass, represents
    the :class:`~asr_eval.align.alignment.Alignment` between ground
    truth and prediction, as well as other useful information.
    """
    ...

    err_positions: dict[asr_eval.align.transcription.OuterLoc, asr_eval.align.alignment.ErrorListingElement]
    """The output of
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """

    metrics: asr_eval.align.metrics.Metrics
    """The output of
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """

    elapsed_time: float
    """Inference time, may be NaN if not known."""

    transcription_html: str | None
    """The aligned transcription in HTML to display."""

    alignment: asr_eval.align.alignment.Alignment
    """The alignment between ground truth and prediction"""