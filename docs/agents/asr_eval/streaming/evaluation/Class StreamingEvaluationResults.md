# Class StreamingEvaluationResults (defined in asr_eval/streaming/evaluation.py at lines 44-99)

@dataclasses.dataclass(kw_only=True)
class StreamingEvaluationResults:
    """A container for evaluation results for a streaming speech
    recognition on a single sample.

    Usually a result of the
    :func:`~asr_eval.streaming.evaluation.evaluate_streaming` function.
    """
    ...

    timed_transcription: asr_eval.align.transcription.Transcription
    """The ground truth transcription for the whole audio with filled
    timings for each token.
    """

    waveform: asr_eval.utils.types.FLOATS
    """A waveform in float32 dtype with sampling rate 16000."""

    cutoffs: list[asr_eval.streaming.sender.Cutoff]
    """A schedule on which the input chunks was sent."""

    input_chunks: list[asr_eval.streaming.model.InputChunk]
    """The input chunks history. The fields :code:`.put_timestamp` and
    :code:`.get_timestamp` are relative to the start time.
    """

    output_chunks: list[asr_eval.streaming.model.OutputChunk]
    """The output chunks history. The fields :code:`.put_timestamp` and
    :code:`.get_timestamp` are relative to the start time.
    """

    partial_alignments: list[asr_eval.streaming.evaluation.PartialAlignment]
    """Alignments of the partial transcriptions against starting parts
    of the ground truth. Each partial alignment keep the
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`
    field that indicates a timestamp relative to the start time.
    """

    @property
    def start_timestamp(self) -> float:
        """A start time, where the first input chunks was put into the
        input buffer.Should be always zero, because all the timestamps
        in the :code:`StreamingEvaluationResults` are relative to this
        moment.
        """
        ...

    @property
    def finish_timestamp(self) -> float:
        """A finish time, where the last output chunk was put into the
        output buffer. The timestamp is relative to the starting
        moment.
        """
        ...