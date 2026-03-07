from __future__ import annotations
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
import re
import string
from typing import Callable

from asr_eval.align.transcription import (
    Wildcard,
    Transcription,
    SingleVariantTranscription,
    Token,
    MultiVariantBlock,
)


__all__ = [
    "Parser",
    "DEFAULT_PARSER",
    "PUNCT",
]


PUNCT = re.escape(r""".,!?:;…-‑–—'"‘“”«»()[]{}""")
r"""
A default set of punctuation characters to exclude from words
:code:`.,!?:;…-‑–—'"‘“”«»()[]{}`. Note that this does not affect parsing
control characters "{", "|", "}" in multivariant syntax. To override,
create a :class:`~asr_eval.align.parsing.Parser` with custom
:attr:`~asr_eval.align.parsing.Parser.tokenizing` field.

:meta hide-value:
"""


@dataclass
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
         'start_time': nan, 'end_time': nan, 'attrs': {}, 'flags': set()}
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
    
    Words (and, in general, tokens) can have attributes
    (:attr:`~asr_eval.align.transcription.Token.attrs`) and flags
    (:attr:`~asr_eval.align.transcription.Token.flags`). They are
    written in brackets, separated by comma from each other, and by "!"
    from the text. For example, the following syntax adds flag "abc"
    and attribute "w" with value "10" to the words bar, baz, qux:
    
    Example:
    
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> text = '"Foo [abc,w=10! bar {baz} qux ]"'
        >>> transcription = DEFAULT_PARSER.parse_transcription(text)
        >>> for block in transcription.blocks: # doctest: +SKIP
        ...     print(block)
        Token(foo)
        Token(bar, attrs:w=10, flags:abc)
        MultiVariantBlock([Token(baz, attrs:w=10, flags:abc)], [])
        Token(qux, attrs:w=10, flags:abc)
    """
    
    tokenizing: str = rf'\w+|[^\w\s{PUNCT}]+'
    r"""A regexp to extract word, by default
    :code:`\\w+|[^\\w\\s{PUNCT}]+`, where
    :const:`~asr_eval.align.parsing.PUNCT` are punctuation
    characters.
       
    :meta hide-value:
    """
    
    preprocessing: Callable[[str], str] = lambda text: text
    """A text preprocessing method set as :code:`Callable[[str], str]`,
    by default does nothing. Is suitable for text-to-text operations
    such as normalizers or filler word removers. Note that after parsing
    the
    :attr:`~asr_eval.align.transcription.Transcription.text` field in
    :class:`~asr_eval.align.transcription.Transcription` will contain
    the preprocessed version, and the original version will be gone.
    
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
    
    postprocessing: Callable[[str], str] = (
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
    ) -> SingleVariantTranscription:
        """Parses a text without multivariant blocks.
        
        In general, one needs this method for typing purposes only,
        because
        :meth:`~asr_eval.align.parsing.Parser.parse_transcription`
        supports both multivariant and single-variant transcriptions.
        """
        text = self.preprocessing(text)
        text, attrs_per_token = _extract_and_trim_attrs(text)
        tokens = self._split_text_into_tokens(text)
        result = SingleVariantTranscription(text, tuple(tokens))
        _assign_attrs_inplace(result, attrs_per_token)
        for i, t in enumerate(result.list_all_tokens()):
            t.uid = 'id' + str(i)
        return result
    
    _MULTIVARIANT_PATTERN = re.compile(
        r'({[^{}]*?})'            # multi variant
        '|'
        r'(?<=})([^{}]+?)(?={)'   # single variant
    )

    # We could also parse multivariant strings with pyparsing:

    # from pyparsing import CharsNotIn, OneOrMore, Suppress as S, Group,
    #     Empty, ZeroOrMore

    # WORDS = CharsNotIn('{|}\n')('words')
    # OPTION = Group(WORDS | Empty())('option')
    # MULTI = Group(S('{') + OPTION + OneOrMore(S('|') + OPTION) \
    #     + S('}'))('multi')
    # MV_STRING = ZeroOrMore(MULTI | WORDS)

    # results = MV_STRING.parse_string('{a|b} ! {1|2 3|} x y {3|4}',
    #     parse_all=True)
    # print(results.as_list())

    # however, this is not obvious for ones who are not familiar with
    # pyparsing, and also gives uninformative parsing errors

    def parse_transcription(self, text: str) -> Transcription:
        """Parses a text possibly containing multivariant blocks.
        
        See example in the class docstring.
        """
        text = self.preprocessing(text)
        text, attrs_per_token = _extract_and_trim_attrs(text)
        blocks: list[Token | MultiVariantBlock] = []
        for match in re.finditer(self._MULTIVARIANT_PATTERN, '}' + text + '{'):
            text_part = match.group()
            start = match.start() - 1  # account for '}' (see in re.finditer)
            end = match.end() - 1      # account for '}' (see in re.finditer)
            if text_part.startswith('{'):
                if start > 0:
                    assert (c := text[start - 1]) in string.whitespace, (
                        f'put a space before a multivariant block, got "{c}"'
                    )
                if end < len(text):
                    assert (c := text[end]) in string.whitespace, (
                        f'put a space after a multivariant block, got "{c}"'
                    )
                
                # options_raw: (option text, start pos)
                options_raw: list[tuple[str, int]] = []
                
                for option_match in re.finditer(
                    r'([^\|]*)\|', text_part[1:-1] + '|'
                ):
                    option_text = option_match.group(1)
                    start_pos = start + option_match.start() + 1
                    if option_text.strip().startswith('~'):
                        # lexically wrong but acceptable form
                        option_text = option_text.strip()[1:]
                    if (match2 := re.match(
                        r'([^\<]+)\<(\w+)\>', option_text.strip()
                    )) is not None:
                        # forms like Facebook<е>
                        # TODO: ambiguous: need to add empty option to
                        # {Facebook<е>} ??
                        # TODO: handle this in single-variant blocks
                        base, suffix = match2.groups()
                        options_raw.append((f'{base}', start_pos))
                        options_raw.append((f'{base}-{suffix}', start_pos))
                    else:
                        options_raw.append((option_text, start_pos))
                
                options: list[list[Token]] = []
                for option_text, start_pos in options_raw:
                    option_tokens = self._split_text_into_tokens(option_text)
                    _shift_tokens_inplace(option_tokens, start_pos)
                    options.append(option_tokens)
                    
                if len(options) == 1:
                    assert len(options[0]), 'empty multivariant block'
                    options.append([])
                
                blocks.append(MultiVariantBlock(
                    options=options,
                    start_pos=start,
                    end_pos=end,
                ))
            else:
                new_tokens = self._split_text_into_tokens(text_part)
                _shift_tokens_inplace(new_tokens, shift=start)
                blocks += new_tokens
        
        result = Transcription(text, tuple(blocks))
        _assign_attrs_inplace(result, attrs_per_token)
        
        for i, t in enumerate(result.list_all_tokens()):
            t.uid = 'id' + str(i)
        
        i = 0
        for block in result.blocks:
            if isinstance(block, MultiVariantBlock):
                block.uid = 'mvid' + str(i)
                i += 1
        
        return result

    def _split_text_into_tokens(self, text: str) -> list[Token]:
        """Finds words in the text and return them as a list of Token.
        """
        tokens = list(_regexp_split_text_into_tokens(
            text, {'word': self.tokenizing}
        ))
        
        for token in tokens:
            if not isinstance(token.value, Wildcard):
                token.value = self.postprocessing(token.value)
        
        return tokens


DEFAULT_PARSER = Parser()
"""An instance of :class:`~asr_eval.align.parsing.Parser` with default
parameters.

:meta hide-value:
"""


def _shift_tokens_inplace(tokens: list[Token], shift: int = 0):
    for t in tokens:
        t.start_pos += shift
        t.end_pos += shift


@lru_cache(maxsize=100)
def _compile_regexp(patterns: tuple[tuple[str, str]]) -> re.Pattern[str]:
    pattern = '|'.join(
        f'(?P<{name}>{subpattern})' for name, subpattern in patterns
    )
    return re.compile(pattern, re.MULTILINE|re.DOTALL|re.UNICODE)


def _regexp_split_text_into_tokens(
    text: str, patterns: dict[str, str]
) -> Iterable[Token]:
    """Searches sequentially for any of the given patterns.
    
    For each match returns a
    :class:`~asr_eval.align.transcription.Token`.
    """
    # this is overcompilcated, TODO simplify?
    pattern = _compile_regexp(tuple(patterns.items()))
    for match in re.finditer(pattern, text):
        found_groups = [
            (name, substr)
            for name, substr in match.groupdict().items()
            if substr is not None
        ]
        assert len(found_groups) == 1
        name, word = found_groups[0]
        assert name in patterns
        
        yield Token(
            value=Wildcard() if word == Wildcard._SYMBOL else word, # pyright: ignore[reportPrivateUsage]
            start_pos=match.start(),
            end_pos=match.end(),
        )


@dataclass(slots=True)
class _MarkedAttr:
    value: str
    start: int
    end: int

_ATTRS_PATTERN = re.compile(
    r'(\[)'  # starting [
    r'([^!\]]*)' # marks
    r'(!)' # ! symbol
    r'([^\]]*)' # transcription
    r'(\])' # ending ]
)

def _extract_and_trim_attrs(text: str) -> tuple[str, list[list[str]]]:

    # we start with this: X = "one two [w=10!Kafka AWS] three"
    # and convert to this: Y = "one two Kafka AWS three"
    # while extracting this: [("w=10", 8, 17)]
    # where (8, 17) is the position of "Kafka AWS" in Y

    # span of value, start symbol, end symbol
    attrs: list[_MarkedAttr] = []

    # iterating while modifying the string
    # TODO speedup this logic for long texts
    while (match := re.search(_ATTRS_PATTERN, text)):
        
        # example: [digits,a=b!три четыре]
        # here: "digits,a=b" - flags for "три четыре"
        
        # extract groups (_ATTRS_PATTERN has 5 groups)
        _, attr, _, _text, _ = match.groups()

        assert attr, f'Attrs should not be empty: {match.group()}'

        # extract group positions
        s1, _s2, s3, _s4, s5 = [match.span(i) for i in range(1, 6)]

        if len(attrs):
            assert s1[0] >= attrs[-1].end, 'Overlapping attrs'

        # spans to cut from text ("[digits,a=b!" and "]" in our example)
        start1, end1 = s1[0], s3[1]
        start2, end2 = s5

        # cut spans from text
        text = text[:start1] + text[end1:start2] + text[end2:]

        # save the attrs
        attr = _MarkedAttr(attr, start1, start1 + start2 - end1)
        attrs.append(attr)

        assert text[attr.start:attr.end] == _text

    attrs_per_token: list[list[str]] = [[] for _ in range(len(text))]
    for attr in attrs:
        for pos in range(attr.start, attr.end):
            for value in attr.value.split(','):
                attrs_per_token[pos].append(value)
    
    return text, attrs_per_token

def _assign_attrs_inplace(
    transcription: Transcription,
    attrs_per_token: list[list[str]]
):
    for token in transcription.list_all_tokens():
        attrs_for_token = Counter(chain(*[
            attrs_per_token[pos]
            for pos in range(token.start_pos, token.end_pos)
        ]))
        for attr, count in attrs_for_token.most_common():
            assert count == token.end_pos - token.start_pos, (
                f'Attr {attr} half-covered a word {token.to_text()}'
            )
            if '=' in attr:
                k, v = attr.split('=', 1)
                token.attrs[k] = v
            else:
                token.flags.add(attr)