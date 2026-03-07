from typing import cast
from dataclasses import dataclass
from itertools import pairwise

from Bio import Align


__all__ = [
    'char_align',
    'CharAligned',
    'get_spans_for_char_aligned',
]


@dataclass
class CharAligned:
    """
    A char-level alignment for two texts, obtained by
    :func:`~asr_eval.align.char_aligner.char_align` (see its dostring
    for details).
    """
    first: str
    matching: str
    second: str


def char_align(
    text_1: str,
    text_2: str,
    placeholder: str = '|',
    ignore_case: bool = True,
) -> CharAligned:
    """A wrapper around *biopython* to perform character-wise alignment.
    
    This algorithm is currently not included in the main *asr_eval*
    workflow, and it does not support multivariant annotation.
    
    The returned dataclass contains 3 strings of equal length, where
    each position represent a match. The first string represents the
    first text, the last string represents the second text, and the
    middle string contains types of maches. For a correct match, the
    second string contains "|" char. For a replacement, the second
    string contains "." char. For deletion or insertion, the second
    string contains "-" char, and the missing character in the first
    or the second text is filled with a :code:`placeholder`.
        
    Args:
        text_1: The first text to align.
        text_2: The second text to align.
        placeholder: A filler for missing (non-matched) characters.
        ignore_case: If True, aligns ignoring case. The output texts
            still contain the original case.
    
    Example:
        >>> al = char_align(
        ...     'Set an alarm for 7 am',
        ...     'SET A ALARM FOR SEVEN A.M.'
        ... )
        >>> print(al.first + '\\n' + al.matching + '\\n' + al.second)
        Set an alarm for 7|||| a|m|
        |||||-|||||||||||.----||-|-
        SET A| ALARM FOR SEVEN A.M.
    """
    
    if not text_1:
        return CharAligned(
            placeholder * len(text_2),
            '-' * len(text_2),
            text_2
        )
    elif not text_2:
        return CharAligned(
            text_1,
            '-' * len(text_1),
            placeholder * len(text_1),
        )
    
    # aligning with replacing dashes (since in PairwiseAligner dash has a
    # special meaning)
    alignment = Align.PairwiseAligner().align( # type: ignore
        (text_1.lower() if ignore_case else text_1).replace('-', '\0'),
        (text_2.lower() if ignore_case else text_2).replace('-', '\0'),
    )[0] # type: ignore
    first, matching, second = (
        cast(str, alignment._format_unicode()) # type: ignore
        .splitlines()
    )
    
    # example intermediate results:
    # first    'ханты0мансийский округ- былино- таурова-'
    # matching '||||.-||.|||||||||||||-||||||.-|||||||.-'
    # second   'ханта-массийский округ, былина, таурово.'
    
    # find where PairwiseAligner put a dash
    placeholder_positions_1 = [
        i for i, char in enumerate(first) if char == '-'
    ]
    placeholder_positions_2 = [
        i for i, char in enumerate(second) if char == '-'
    ]
    
    # restoring the original chars for both texts
    first = text_1
    for pos in placeholder_positions_1:
        first = first[:pos] + placeholder + first[pos:]
    second = text_2
    for pos in placeholder_positions_2:
        second = second[:pos] + placeholder + second[pos:]
    
    # example results:
    # first    'ханты-мансийский округ| былино| таурова|'
    # matching '||||.-||.|||||||||||||-||||||.-|||||||.-'
    # second   'Ханта|массийский округ, Былина, Таурово.'
    
    return CharAligned(first, matching, second)

def get_spans_for_char_aligned(
    al: CharAligned
) -> list[tuple[int, int]]:
    """
    Locates positions in the
    :class:`~asr_eval.align.char_aligner.CharAligned` where both texts
    contain space. Splits by these positions, and returns the resulting
    spans.
    
    This can be useful to match predictions from two models, where
    there may be cases when a single word from one first model matches
    with two words from another model. In this case the current function
    will return an interval spanning both words.
    """
    segment_sep_positions: list[int] = []
    for char_idx, (first_char, alignment_notation, second_char) in (
        enumerate(zip(al.first, al.matching, al.second, strict=True))
    ):
        if first_char == ' ' and second_char == ' ':
            assert alignment_notation == '|'
            segment_sep_positions.append(char_idx)
            
    # we cannot just prepend 0 because this index can be already in
    # segment_sep_positions
    segment_sep_positions = (
        sorted(set(segment_sep_positions) | {0, len(al.first)})
    )
    
    results: list[tuple[int, int]] = []
    for start_char_pos, end_char_pos in pairwise(segment_sep_positions):
        if start_char_pos != 0:
            start_char_pos += 1  # omit space
        if end_char_pos > start_char_pos:
            results.append((start_char_pos, end_char_pos))
    
    return results