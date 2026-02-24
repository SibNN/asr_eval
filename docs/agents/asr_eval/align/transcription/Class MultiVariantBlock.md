# Class MultiVariantBlock (defined in asr_eval/align/transcription.py at lines 127-196)

@dataclasses.dataclass(slots=True)
class MultiVariantBlock:
    """
    A multivariant block in a transcription. Contains two or more
    options, each option contains zero or more tokens.
    """
    ...

    options: list[list[asr_eval.align.transcription.Token]]

    start_pos: int = 0
    """Start of the span in the original text, including braces {}."""

    end_pos: int = 0
    """End of the span the original text, including braces {}."""

    uid: asr_eval.align.transcription.TOKEN_UID = field(default_factory=new_uid)
    """An ID that is unique in the text. Is used to refer the block."""

    @property
    def is_timed(self) -> bool:
        """True if :attr:`~asr_eval.align.transcription.Token.is_timed`
        is true for all tokens in the block.
        """
        ...

    @property
    def start_time(self) -> float:
        """
        The earliest
        :attr:`~asr_eval.align.transcription.Token.start_time` across
        all options, or NaN if tokens are not timed.
        """
        ...

    @property
    def end_time(self) -> float:
        """
        The latest :attr:`~asr_eval.align.transcription.Token.end_time`
        across all options, or NaN if tokens are not timed.
        """
        ...

    def get_option_text(self, option_index: int) -> str:
        """For displaying purposes."""
        ...

    def to_text(self) -> str:
        """For displaying purposes."""
        ...