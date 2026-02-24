# Class Parser (defined in asr_eval/align/parsing.py at lines 36-289)

@dataclasses.dataclass
class Parser:
    r"""Parses into words and (optionally) normalizes prediction or
    annotation.

    Performs the following:

    1. Preprocesses the whole text if
       :attr:`~asr_eval.align.parsing.Parser.preprocessing` is set. This
       stage is suitable for various normalization methods, if they are
       used, such as numerals-to-digits normalizers or filler words
       removers.
    2. If
       :meth:`~asr_eval.align.parsing.Parser.parse_transcription`,
       is called, processes multivariant syntax.
    3. Splits all the text blocks into words with a regexp stored in
       the :attr:`~asr_eval.align.parsing.Parser.tokenizing` attribute.
    4. Postprocesses each word if
       :attr:`~asr_eval.align.parsing.Parser.postprocessing` is set.
       This stage is suitable for lowercase conversion.

    A :const:`~asr_eval.align.parsing.DEFAULT_PARSER` is an instance
    of the Parser with default parameters.

    Example:
        >>> from asr_eval.align.parsing import DEFAULT_PARSER  # same as Parser()
        >>> text = 'Hi there {fouth|4|t-th} {eh} <*>'
        >>> parsed = DEFAULT_PARSER.parse_transcription(text)
        >>> print(parsed.blocks) # doctest: +NORMALIZE_WHITESPACE
        (Token(hi),
         Token(there),
         MultiVariantBlock([Token(fouth)], [Token(4)], [Token(t), Token(th)]),
         MultiVariantBlock([Token(eh)], []),
         Token(Wildcard()))
        >>> from dataclasses import asdict
        >>> asdict(parsed.blocks[0]) # doctest: +NORMALIZE_WHITESPACE
        {'value': 'hi', 'uid': 'id0', 'start_pos': 0, 'end_pos': 2,
         'start_time': nan, 'end_time': nan}
        >>> print(parsed.colorize()) # doctest: +SKIP

    .. raw:: html

        <style>.y {background-color: #e0e841;} .g {background-color: #41e8a8;} .b {background-color: #41c7e8   ;}</style>
        <span style="white-space='pre'; font-family: 'Consolas', 'Ubuntu Mono', 'Monaco', monospace">
        <span class="y">Hi</span> <span class="g">there</span>
        {<span class="b">fourth</span>|<span class="y">4</span>|<span class="g">4</span>-<span class="b">th</span>}
        {<span class="y">eh</span>} <span class="g"><*></span></span>

    Note:
        1. Why not just :code:`nltk.word_tokenize`? In *asr_eval* words
           keep references to their positions in the original text,
           which :code:`word_tokenize` does not support.
        2. By making a Parser with
           :code:`tokenizing=r'\\w|\\s|[^\\w\\s{PUNCT}]'` you can parse
           strings into characters, excluding punctuation. In this case,
           :class:`~asr_eval.align.alignment.Alignment` will calculate
           CER (character error rate) instead of WER.
        3. You can create named parsers in
           :mod:`asr_eval.bench.parsers`.
        4. When labeling a dataset, the annotator should be aware of the
           tokenization scheme. For example, if :code:`3/4$` is
           tokenized as a single word, then :code:`3/4$` and
           :code:`3 / 4 $` (with spaces) are different options, and both
           should be included in a multivariant block. See
           :doc:`/guide_alignment_wer` for details.
    """
    ...

    tokenizing: str = rf'\w+|[^\w\s{PUNCT}]+'
    r"""A regexp to extract word, by default
    :code:`\\w+|[^\\w\\s{PUNCT}]+`, where
    :const:`~asr_eval.align.parsing.PUNCT` are punctuation
    characters.

    :meta hide-value:
    """

    preprocessing: typing.Callable[[str], str] = lambda text: text
    """A text preprocessing method set as :code:`Callable[[str], str]`,
    by default does nothing. Is suitable for text-to-text operations
    such as normalizers or filler word removers. Note that after parsing
    the
    :attr:`~asr_eval.align.transcription.Transcription.text` field in
    :class:`~asr_eval.align.transcription.Transcription` willcontain the
    preprocessed version, and the original version will be gone.

    Example:
        >>> from asr_eval.align.parsing import Parser
        >>> import re
        >>> def filler_remover(text: str) -> str:
        ...     for word in 'eh', 'oh', 'umm':
        ...         text = re.sub(word, '', text, flags=re.IGNORECASE)
        ...     return text
        >>> parser = Parser(preprocessing=filler_remover)
        >>> parsed = parser.parse_transcription('Umm eh of course')
        >>> print(parsed.text, parsed.blocks)
        of course [Token(of), Token(course)]

    See more examples in :mod:`asr_eval.bench.parsers`.

    :meta hide-value:
    """

    postprocessing: typing.Callable[[str], str] = (
        lambda text: text.lower().replace('ё', 'е')
    )
    """
    A word postprocessing method set as :code:`Callable[[str], str]`,
    by default performs lowercase and diacritic conversion:

    .. code-block:: python

        postprocessing=lambda text: text.lower().replace('ё', 'е')

    Will only affect the
    :attr:`~asr_eval.align.transcription.Token.value` field in
    :class:`~asr_eval.align.transcription.Token`. This is useful to
    match lowercase words, while tracking their positions in
    the original
    :attr:`~asr_eval.align.transcription.Transcription.text` with
    capitalization and punctuation.

    :meta hide-value:
    """

    def parse_single_variant_transcription(
        self, text: str
    ) -> asr_eval.align.transcription.SingleVariantTranscription:
        """Parses a text without multivariant blocks.

        In general, one needs this method for typing purposes only,
        because
        :meth:`~asr_eval.align.parsing.Parser.parse_transcription`
        supports both multivariant and single-variant transcriptions.
        """
        ...

    def parse_transcription(self, text: str) -> asr_eval.align.transcription.Transcription:
        """Parses a text possibly containing multivariant blocks.

        See example in the class docstring.
        """
        ...