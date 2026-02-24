# Class Token (defined in asr_eval/align/transcription.py at lines 44-125)

@dataclasses.dataclass(slots=True)
class Token:
    """Represents typically a single word or a
    :class:`~asr_eval.align.transcription.Wildcard` symbol in a
    transcription.

    Note:
        Typically you don't need to create Token manually. They are created
        automatically when parsing a text with
        :class:`~asr_eval.align.parsing.Parser`.

    Token is what the alignment algorithm considers an atom. Two tokens
    may be equal or not. If not equal, this contributes to the error
    count. By default :class:`~asr_eval.align.parsing.Parser` splits a
    text into words, and in this case each token is a words, and we
    obtain WER (word error rate) by aligning them. However, with
    modified :attr:`~asr_eval.align.parsing.Parser.tokenizing` regexp
    a parser may split into characters - then each token stores a single
    character. See the :class:`~asr_eval.align.parsing.Parser` docs for
    defails. There is also a special token with :code:`.value=Wilcard()`
    that matches every token equence, possibly empty.
    """
    ...

    value: str | asr_eval.align.transcription.Wildcard
    """A text (usually a word) after all the normalization steps, or
    a :class:`~asr_eval.align.transcription.Wildcard` symbol. If not
    wilcard, two tokens match if their texts are equal as strings.
    """

    uid: asr_eval.align.transcription.TOKEN_UID = field(default_factory=new_uid)
    """An ID that is unique in the text. Is used to refer the token."""

    start_pos: int = 0
    """The start position in the original text (the :code:`.text` field
    of the :class:`~asr_eval.align.transcription.Transcription` the
    token belongs to).
    """

    end_pos: int = 0
    """The end position in the original text (the :code:`.text` field
    of the :class:`~asr_eval.align.transcription.Transcription` the
    token belongs to), not inclusive.
    """

    start_time: float = np.nan
    """The start time in seconds, is NaN by default, can be filled by
    :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
    """

    end_time: float = np.nan
    """The end time in seconds, is NaN by default, can be filled by
    :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
    """

    @property
    def is_timed(self) -> bool:
        """Are :attr:`~asr_eval.align.transcription.Token.start_time`
        and :attr:`~asr_eval.align.transcription.Token.end_time` not
        NaN?
        """
        ...

    def to_text(self) -> str:
        """Returns :code:`.value`, or :code:`<*>` if Wildcard, for
        displaying purposes.
        """
        ...