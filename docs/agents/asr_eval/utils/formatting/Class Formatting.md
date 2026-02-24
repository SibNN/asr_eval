# Class Formatting (defined in asr_eval/utils/formatting.py at lines 19-39)

@dataclasses.dataclass
class Formatting:
    """ ANSI text formatting attrubutes, such as "bold", "red" etc.

    Example:
        >>> from asr_eval.utils.formatting import Formatting
        >>> Formatting(color='red', attrs={'strike'}) # doctest: +ELLIPSIS
        ...
    """
    ...

    color: str | None = None

    on_color: str | None = None

    attrs: set[str] = field(default_factory=set[str])