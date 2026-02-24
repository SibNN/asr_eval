# Class Wildcard (defined in asr_eval/align/transcription.py at lines 31-42)

@dataclasses.dataclass(slots=True)
class Wildcard:
    """
    Represents a Wilcard symbol <*> in a transcription that matches
    every word sequence, possibly empty.
    """
    ...