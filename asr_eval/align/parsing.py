from __future__ import annotations

import re
import string

from .data import Anything, Token, MultiVariant
from ..utils import apply_ansi_formatting, Formatting, FormattingSpan

# equals nltk.WordPunctTokenizer()._regexp used for nltk.wordpunct_tokenize(text)
# we cannot use nltk.wordpunct_tokenize because we also need word spans (start, end)
WORD_REGEXP = re.compile(r'\w+|[^\w\s]+', re.MULTILINE|re.DOTALL|re.UNICODE)

def split_text_into_tokens(
    text: str,
    pos_shift: int = 0,
) -> list[Token]:
    result: list[Token] = []
    for match in re.finditer(WORD_REGEXP, text):
        word = match.group()
        start, end = match.span()
        result.append(Token(
            value=word if word != '<*>' else Anything(),
            start_pos=start + pos_shift,
            end_pos=end + pos_shift,
        ))
    return result


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
) -> list[Token | MultiVariant]:
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
            options = [
                split_text_into_tokens(option.group(1), pos_shift=start + option.start() + 1)
                for option in re.finditer(r'([^\|]*)\|', text_part[1:-1] + '|')
            ]
            if len(options) == 1:
                options.append([])
            result.append(MultiVariant(options=options, pos=(start, end)))
        else:
            result += split_text_into_tokens(text_part, pos_shift=start)
    return result


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