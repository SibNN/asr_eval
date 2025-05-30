from __future__ import annotations

import re
import string

from .data import Anything, Token, MultiVariant


def parse_string(
    text: str,
    pos_shift: int = 0,
) -> list[Token]:
    result: list[Token] = []
    for match in re.finditer(r'\S+', text):
        word = match.group()
        start, end = match.span()
        result.append(Token(
            value=word if word != '<*>' else Anything(),
            pos=(start + pos_shift, end + pos_shift),
        ))
    return result

MULTIVARIANT_PATTERN = re.compile(
    r'({[^{}]*?})'            # multi variant
    '|'
    r'(?<=})([^{}]+?)(?={)'   # single variant
)


def parse_multi_variant_string(
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
            result.append(MultiVariant(
                options=[
                    parse_string(option.group(1), pos_shift=start + option.start() + 1)
                    for option in re.finditer(r'([^\|]*)\|', text_part[1:-1] + '|')
                ],
                pos=(start, end),
            ))
        else:
            result += parse_string(text_part, pos_shift=start)
    return result