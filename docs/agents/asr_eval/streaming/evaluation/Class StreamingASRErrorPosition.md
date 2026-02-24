# Class StreamingASRErrorPosition (defined in asr_eval/streaming/evaluation.py at lines 375-425)

@dataclasses.dataclass
class StreamingASRErrorPosition:
    """A word-level match in a
    :class:`~asr_eval.streaming.evaluation.PartialAlignment` with
    assigned
    :attr:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition.status`.
    """
    ...

    start_time: float
    """Start time of the ground truth word. If the match is insertion,
    no ground truth word exists, and the start time is the end time of
    the previous ground truth word, or zero.
    """

    end_time: float
    """End time of the ground truth word. If the match is insertion,
    no ground truth word exists, and the end time is the start time of
    the next ground truth word, or the processed time.
    """

    sent_time: float
    """How much seconds of the input audio was sent at the time when
    the current partial alignment was calculated.
    """

    processed_time: float
    """How much seconds of the input audio was processed at the time
    when the current partial alignment was calculated (see
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`).
    """

    status: (
        typing.Literal['correct', 'deletion', 'insertion', 'replacement', 'not_yet']
    )
    """One of 5 statuses: ("correct", "deletion", "insertion",
    "replacement", "not_yet"). The first 4 statuses are explained
    in the :attr:`~asr_eval.align.matching.Match.status`. The status
    "not_yet" is a special status that is assigned for trailing
    deletions. We consider that if a deletion is trailing, it represents
    a word not transcribed yet. This may occur either due to long
    inference times which cause delays, or because a model refuses to
    transcribe until it accumulates enough context. The field
    :attr:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition.processed_time`
    allows to differentiate between these two reasons.
    """

    @property
    def center_time(self) -> float:
        """A center between the start and the end time."""
        ...