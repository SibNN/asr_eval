# Class WordCorrection (defined in asr_eval/correction/corrector_levenshtein.py at lines 21-30)

@dataclasses.dataclass
class WordCorrection:
    """A suggestion to replace :code:`text[start:end]` with
    :code:`correction`.
    """
    ...

    start: int

    end: int

    correction: str