from __future__ import annotations

from dataclasses import replace
import re
import string
from typing import Literal, cast

import razdel

from .data import Anything, Token, MultiVariant
from ..utils.utils import apply_ansi_formatting, Formatting, FormattingSpan
    

# def _strip_and_locate(text: str, strip: str) -> tuple[str, int, int]:
#     '''
#     strips a word, return 1) the result, 2) start delta, 3) end delta (negative)
#     '''
#     stripped = text.strip(strip)
#     d_start = text.find(stripped)  # will be 0 if stripped is empty
#     d_end = -(len(text) - d_start - len(stripped))
#     return stripped, d_start, d_end


def razdel_split_text_into_tokens(text: str) -> list[Token]:
    tokens: list[Token] = []
    for word in razdel.tokenize(text):
        value = cast(str, word.text) # pyright:ignore[reportUnknownMemberType]
        start = cast(int, word.start) # pyright:ignore[reportUnknownMemberType]
        stop = cast(int, word.stop) # pyright:ignore[reportUnknownMemberType]
        tokens.append(Token(
            value=value,
            start_pos=start,
            end_pos=stop,
        ))
    while True:
        for i in range(len(tokens) - 2):
            if (
                tokens[i + 2].end_pos == tokens[i].start_pos + 3
                and tokens[i].value == '<'
                and tokens[i + 1].value == '*'
                and tokens[i + 2].value == '>'
            ):
                tokens = (
                    tokens[:i]
                    + [Token(
                        value=Anything(),
                        start_pos=tokens[i].start_pos,
                        end_pos=tokens[i].start_pos + 3
                    )]
                    + tokens[i + 3:]
                )
                break
        else:
            break
    
    return tokens


def regexp_split_text_into_tokens(text: str, patterns: dict[str, str]):
    tokens: list[Token] = []
    pattern = '|'.join(f'(?P<{name}>{subpattern})' for name, subpattern in patterns.items())
    for match in re.finditer(re.compile(pattern, re.MULTILINE|re.DOTALL|re.UNICODE), text):
        found_groups = [
            (name, substr)
            for name, substr in match.groupdict().items()
            if substr is not None
        ]
        assert len(found_groups) == 1
        name, word = found_groups[0]
        assert name in patterns
        
        tokens.append(Token(
            value=Anything() if word == '<*>' else word,
            start_pos=match.start(),
            end_pos=match.end(),
            type=name,
        ))
    return tokens


def split_text_into_tokens(
    text: str,
    method: Literal['wordpunct_tokenize', 'space', 'razdel', 'asr_eval'] = 'asr_eval',
    drop_punct: bool = True,
    lower: bool = True,
    ru_tweaks: bool = True,
) -> list[Token]:
    """
    Finds words in the text and return them as a list of Token.  For "method" see
    `parse_multivariant_string` docstring.
    """
    if method == 'razdel':
        tokens = razdel_split_text_into_tokens(text)
    else:
        match method:
            case 'wordpunct_tokenize':
                options = {
                    'word': r'\w+',
                    'punct': r'[^\w\s]+',
                }
            case 'space':
                options = {
                    'word': r'\S+',
                }
            case 'asr_eval':
                punct = re.escape(r'''.,!?:;…-–—'"‘“”()[]{}''')
                options = {
                    'word': rf'\w+|[^\w\s{punct}]+',
                    'punct': rf'[{punct}]+',
                }
        if drop_punct:
            options.pop('punct', None)
        tokens = regexp_split_text_into_tokens(text, options)
    
    for token in tokens:
        if not isinstance(token.value, Anything):
            if lower:
                token.value = token.value.lower()
            if ru_tweaks:
                token.value = token.value.replace('ё', 'е')
    
    return tokens


# We can also parse multivariant strings with pyparsing:

# from pyparsing import CharsNotIn, OneOrMore, Suppress as S, Group, Empty, ZeroOrMore

# WORDS = CharsNotIn('{|}\n')('words')
# OPTION = Group(WORDS | Empty())('option')
# MULTI = Group(S('{') + OPTION + OneOrMore(S('|') + OPTION) + S('}'))('multi')
# MV_STRING = ZeroOrMore(MULTI | WORDS)

# results = MV_STRING.parse_string('{a|b} ! {1|2 3|} x y {3|4}', parse_all=True)
# print(results.as_list())

# however, this is not obvious for ones who are not familiar with pyparsing
# and also gives uninformative parsing errors


MULTIVARIANT_PATTERN = re.compile(
    r'({[^{}]*?})'            # multi variant
    '|'
    r'(?<=})([^{}]+?)(?={)'   # single variant
)


def parse_multivariant_string(
    text: str,
    method: Literal['wordpunct_tokenize', 'space', 'razdel', 'asr_eval'] = 'asr_eval',
    drop_punct: bool = True,
    lower: bool = True,
    ru_tweaks: bool = True,
) -> list[Token | MultiVariant]:
    r"""
    Finds words in the text, possibly with multivariant blocks, and return them as a list of
    Token and/or MultiVariant objects.
    
    The method "razdel" uses a custom algorithm. The other methods return Token object
    with .type field filled with one of "word" or "punct".
    
    - The method "wordpunct_tokenize" treat \w+ as word and [^\w\s]+ as punct (this equals
    the regexp used in nltk.wordpunct_tokenize).
    - The method "asr_eval" treat \w+ or [^\w\s{P}]+ as word and [{P}]+ as punct, where {P}
    is one of the following symbols: .,!?:;…-–—'"‘“”()[]{}
    - The method "space" treat \S+ as word and nothing as punct.
    
    `drop_punct` removes "punct" tokens (not supported for razdel).
    
    NOTE: An annotator should know how such a postprocessing works. For example, if "3/4$" is
    treated as a single word, then "3/4$" and "3 / 4 $" are different options: if we annotate
    only "3/4$", then a prediction "3 / 4 $" will be considered as 4 errors. Razdel has a
    complex tokenization rules and hence is not suitable (an annotator should know clearly how
    does it work). On the other side, if we split by space, we need to enumerate a lot of
    options for "3/4$!".
    
    Possible problems with "asr_eval" method:
    1) If a speaker speaks "point" or "semicolon", how do we evaluate an ASR prediction?
    2) If "16-й" is a correct transcription, then "16 й" will always be considered correct.
    
    Example:
    ```
    from asr_eval.align.parsing import parse_multivariant_string, colorize_parsed_string
    
    text = '7-8 мая (в Пуэрто-Рико) прошел {шестнадцатый | 16-й | 16} этап "Формулы-1" с фондом 100,000$!'

    for method in 'razdel', 'wordpunct_tokenize', 'space', 'asr_eval':
        tokens = parse_multivariant_string(text, method=method)
        colored_str, colors = colorize_parsed_string(text, tokens)
        print(f'{method: <20}', colored_str)
    print()
    ```
    """
    result: list[Token | MultiVariant] = []
    for match in re.finditer(MULTIVARIANT_PATTERN, '}' + text + '{'):
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
            result.append(MultiVariant(
                options=[
                    _shift_tokens(
                        split_text_into_tokens(
                            option.group(1),
                            method=method,
                            drop_punct=drop_punct, 
                            lower=lower,
                            ru_tweaks=ru_tweaks,
                        ),
                        shift=start + option.start() + 1
                    )
                    for option in re.finditer(r'([^\|]*)\|', text_part[1:-1] + '|')
                ],
                pos=(start, end),
            ))
        else:
            result += _shift_tokens(
                split_text_into_tokens(
                    text_part,
                    method=method,
                    drop_punct=drop_punct, 
                    lower=lower,
                    ru_tweaks=ru_tweaks,
                ),
                shift=start
            )
    
    return result


def _shift_tokens(tokens: list[Token], shift: int = 0) -> list[Token]:
    return [
        replace(t, start_pos=t.start_pos + shift, end_pos=t.end_pos + shift)
        for t in tokens
    ]


def colorize_parsed_string(
    text: str, tokens: list[Token | MultiVariant]
) -> tuple[str, dict[str, str]]:
    '''
    Colorizes each token in the parsed (possibly multivariant) string. Returns:
    - String with ANSI escape codes (can be rendered using print() in jupyter or console).
    - A mapping from token uid to its color.
    
    Example:
    
    ```
    orig_text = '{emm} at {fourty nine|49} <*> year'
    tokens = parse_multivariant_string(orig_text)
    colored_str, colors = colorize_parsed_string(orig_text, tokens)
    print(colored_str)
    ```
    '''
    colors = [
        'on_light_yellow',
        'on_light_green',
        'on_light_cyan',
        # 'on_light_grey',
        # 'on_light_red',
        # 'on_light_blue',
        # 'on_light_magenta',
    ]
    
    token_idx = 0
    token_uid_to_color: dict[str, str] = {}
    
    formatting_spans = [FormattingSpan(
        Formatting(attrs={'bold'}), 0, len(text)
    )]
    
    def mark_token(token: Token):
        nonlocal token_idx, token_uid_to_color
        color = colors[token_idx % len(colors)]
        formatting_spans.append(FormattingSpan(
            Formatting(on_color=color), token.start_pos, token.end_pos,
        ))
        token_uid_to_color[token.uid] = color
        token_idx += 1
        
    for block in tokens:
        match block:
            case Token():
                mark_token(block)
            case MultiVariant():
                formatting_spans.append(FormattingSpan(
                    Formatting(attrs={'underline'}), block.pos[0], block.pos[1]
                ))
                for option in block.options:
                    for token in option:
                        mark_token(token)

    return apply_ansi_formatting(text, formatting_spans), token_uid_to_color