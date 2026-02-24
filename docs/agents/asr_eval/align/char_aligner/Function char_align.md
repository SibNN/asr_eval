# Function char_align (defined in asr_eval/align/char_aligner.py at lines 27-117)

def char_align(
    text_1: str,
    text_2: str,
    placeholder: str = '|',
    ignore_case: bool = True,
) -> asr_eval.align.char_aligner.CharAligned:
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
    ...