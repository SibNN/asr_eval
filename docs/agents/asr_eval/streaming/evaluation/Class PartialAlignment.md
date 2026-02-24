# Class PartialAlignment (defined in asr_eval/streaming/evaluation.py at lines 254-373)

@dataclasses.dataclass
class PartialAlignment:
    """
    An alignment between the ground truth up to the
    :code:`audio_seconds_sent` and the partial transcription.
    """
    ...

    pred: asr_eval.align.transcription.SingleVariantTranscription
    """A partial transcription from the streaming model. While the raw
    transcription is provided in form of the transcription chunks, this
    field represents the chunks joined with
    :meth:`~asr_eval.streaming.model.TranscriptionChunk.join` to form
    a transcription as text, and then parsed into words.
    """

    alignment: asr_eval.align.matching.MatchesList
    """An alignment between the ground truth starting part, and the
    partial transcription.
    """

    at_time: float
    """The timestamp where the alignment was evaluated. All the output
    chunks sent later than this timestamp are not included.
    """

    audio_seconds_sent: float
    """How many seconds of the audio was sent by the time
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`.
    """

    audio_seconds_processed: float
    """How many seconds of the audio was processed by the time
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`.
    This value is extracted from the output chunks (see
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`).
    """

    def get_error_positions(self) -> list[asr_eval.streaming.evaluation.StreamingASRErrorPosition]:
        """Categorizes each word match from
        :attr:`~asr_eval.streaming.evaluation.PartialAlignment.alignment`
        into one of 5 types: ("correct", "deletion", "insertion",
        "replacement", "not_yet"). See the
        :class:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition`
        docs for details.
        """
        ...