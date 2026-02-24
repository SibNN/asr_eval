# Class Transcription (defined in asr_eval/align/transcription.py at lines 198-433)

@dataclasses.dataclass(frozen=True)
class Transcription:
    """
    A transcription that is normalized and parsed into words.

    Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_transcription`.
    May contain zero or more multivariant blocks or
    :class:`~asr_eval.align.transcription.Wildcard` insertions. Stores
    `.text` field (the original text) and list of tokens (words) that
    keep references to the positions in the original text.
    A generic for two transcription subclasses:

    Has a subclass
    :class:`~asr_eval.align.transcription.SingleVariantTranscription`
    used for typing clarity where we do not expect multivariance (mainly
    for predictions). Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_single_variant_transcription`.

    Example:

        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> transcription = DEFAULT_PARSER.parse_transcription(
        ...     "Alexa skip to friday {skip to friday} "
        ...     ", {Don't|Do not} need another sad day <*>"
        ... )
        >>> print(transcription.blocks) # doctest: +NORMALIZE_WHITESPACE
        (Token(alexa),
         Token(skip),
         Token(to),
         Token(friday),
         MultiVariantBlock([Token(skip), Token(to), Token(friday)], []),
         MultiVariantBlock([Token(don), Token(t)], [Token(do), Token(not)]),
         Token(need),
         Token(another),
         Token(sad),
         Token(day),
         Token(Wildcard()))

         >>> from IPython.display import HTML  # doctest: +SKIP
         >>> HTML(transcription.colorize(color_mode='html')) # doctest: +SKIP

        .. raw:: html

            <iframe src="_static/transcription_docstring.html"
                style="border: none; width: 100%; height: 50px; overflow: hidden;"></iframe>
    """
    ...

    text: str
    """The original text that was parsed into words."""

    blocks: tuple[asr_eval.align.transcription.Token | asr_eval.align.transcription.MultiVariantBlock, ...]
    """For single-varian transcription is a list of
    :class:`~asr_eval.align.transcription.Token`. For multivariant
    transcription may also contain zero or more
    :class:`~asr_eval.align.transcription.MultiVariantBlock`.
    """

    def list_all_tokens(self) -> collections.abc.Iterator[asr_eval.align.transcription.Token]:
        """Iterates over all the tokens, including ones in multivariant
        blocks.
        """
        ...

    def is_timed(self) -> bool:
        """A transcription can become timed, if we fill time (in seconds)
        for all the words. This can be done with
        :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        Otherwise :code:`is_timed` is False.
        """
        ...

    def get_starting_part(self, time: float) -> asr_eval.align.transcription.Transcription:
        """ Cut a timed transcription up to the specified time. Is
        primarily used for streaming evaluation of partial
        transcriptions.

        A transcription is timed if all the tokens have their
        :attr:`~asr_eval.align.transcription.Token.start_time` and
        :attr:`~asr_eval.align.transcription.Token.end_time` filled with
        not-NaN values. The current method selects only the tokens
        up to the specified time.

        If :code:`time` is inside a token, converts it into a
        multivariant block with options :code:`[token]` and :code:`[]`.
        For example, let :code:`blocks = [A, B]`, token :code:`A` spans
        from 1.0 to 2.0 and :code:`B` spans from 3.0 to 4.0. Then
        :code:`get_starting_part(time=3.5)` returns
        :code:`[A, MultiVariant(X)]`, where :code:`X == [[B], []]`.

        If :code:`time` is inside an existing multivariant block, then
        cuts each option up to the :code:`time`, and if :code:`time` is
        inside some token in some option, add another option with this
        token excluded. For example, let
        :code:`blocks = [A, MultiVariant([[B1], [B2, B3]])]`, and
        :code:`B1` spans from 3.0 to 4.0, :code:`B2` spans from 3.0 to
        3.5, :code:`B3` spans from 3.5 to 4.0. Then
        :code:`get_starting_part(time=3.7)` returns
        :code:`[A, MultiVariant(X)]`, where
        :code:`X == [[], [B1], [B2], [B2, B3]]`. Here :code:`[]` was
        obtained from cutting option :code:`[B1]` and :code:`[B2]` was
        obtained from cutting option :code:`[B2, B3]`.

        Returns a copy without modifying the original object.
        """
        ...

    def colorize(self,  color_mode: typing.Literal['ansi', 'html'] = 'ansi') -> str:
        """Colorizes each token in the parsed (possibly multivariant)
        string. Returns string with ANSI escape codes (rendered using
        :code:`print` in jupyter or console), or HTML color spans.

        See example in the :class:`~asr_eval.align.alignment.Alignment`
        docstring.
        """
        ...

    def select_single_path(
        self, multivariant_choices: collections.abc.Sequence[int]
    ) -> asr_eval.align.transcription.TranscriptionPath:
        """Returns a transcription with the selected option in each
        multivariant block.

        Note:
            This is a lower-level function typically not called manually
            use :func:`~asr_eval.align.alignment.Alignment` constructor
            instead.

        The :code:`multivariant_choices` are usually obtained from
        :func:`~asr_eval.align.solvers.dynprog.solve_optimal_alignment`.
        The :code:`multivariant_choices` length should equal the total
        count of multivariant blocks.
        """
        ...