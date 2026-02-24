# Class TimedText (defined in asr_eval/segments/segment.py at lines 99-106)

@dataclasses.dataclass(frozen=True)
class TimedText(asr_eval.segments.segment.AudioSegment):
    """An :class:`~asr_eval.segments.AudioSegment` with the
    corresponding text.
    """
    ...

    text: str