# Function split_text_into_sentences (defined in asr_eval/linguistics/linguistics.py at lines 153-195)

def split_text_into_sentences(
    text: str,
    language: typing.Literal['russian', 'english'] = 'russian',
    max_symbols: int | None = None,
    merge_smaller_than: int | None = None,
) -> list[str]:
    """Split the text into sentences using nltk.

    If some sentence has more than :code:`max_symbols` symbols, will
    split it further by space symbols so that each part has no more than
    :code:`max_symbols` symbols. If a single word has more than
    :code:`max_symbols` symbols, it will be kept as is (no truncation or
    dividing a word into parts).

    If :code:`merge_smaller_than` is specified, tries to merge sentences
    smaller than the specified value, without exceeding
    :code:`max_symbols`.
    """
    ...