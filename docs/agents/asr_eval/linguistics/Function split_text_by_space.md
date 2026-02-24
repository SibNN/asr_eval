# Function split_text_by_space (defined in asr_eval/linguistics/linguistics.py at lines 197-219)

def split_text_by_space(text: str, max_symbols: int) -> list[str]:
    r"""Split text into parts by space (\s) symbols so that each part
    has no more than :code:`max_symbols` symbols. If a single word has
    more than :code:`max_symbols` symbols, it will be kept as is (no
    truncation or dividing a word into parts).
    """
    ...