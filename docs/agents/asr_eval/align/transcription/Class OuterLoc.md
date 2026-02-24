# Class OuterLoc (defined in asr_eval/align/transcription.py at lines 446-456)

@dataclasses.dataclass(frozen=True)
class OuterLoc:
    """A slot that represents a specific position in the ground truth:
    before/at/after some word index.

    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    ...

    mod: typing.Literal['at', 'pre']

    pos: int