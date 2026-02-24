# Class InnerLoc (defined in asr_eval/align/transcription.py at lines 457-468)

@dataclasses.dataclass(frozen=True)
class InnerLoc(asr_eval.align.transcription.OuterLoc):
    """A slot that represents a specific inner position in the
    multivariant ground truth: before/at/after some word index
    in a multivariant option.

    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    ...

    inner_mod: typing.Literal['at', 'pre']

    inner_pos: int