# Class DiarizationSegment (defined in asr_eval/segments/segment.py at lines 108-115)

@dataclasses.dataclass(frozen=True)
class DiarizationSegment(asr_eval.segments.segment.AudioSegment):
    """An :class:`~asr_eval.segments.AudioSegment` with the
    corresponding speaker index or name.
    """
    ...

    speaker: int | str