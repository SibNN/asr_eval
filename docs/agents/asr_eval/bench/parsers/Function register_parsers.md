# Function register_parsers (defined in asr_eval/bench/parsers/_registry.py at lines 13-42)

def register_parsers(
    name: str, true_parser: type[asr_eval.align.parsing.Parser], pred_parser: type[asr_eval.align.parsing.Parser]
):
    r"""Register a pair of parsers: one for the annotation and another
    for the prediction. To specify a custom parser, you need to
    subclass the :class:`~asr_eval.align.parsing.Parser` class so that
    the constructor does not accept arguments, and register it here.

    Example:
        >>> # we will register a new char-wise parser
        >>> from asr_eval.align.parsing import PUNCT, Parser
        >>> from asr_eval.bench.parsers import register_parsers
        >>> from asr_eval.bench.parsers._registry import get_parser
        >>> class CharWiseParser(Parser):
        ...     def __init__(self):
        ...         super().__init__(tokenizing=rf'[^\s{PUNCT}]')
        >>> register_parsers('charwise', CharWiseParser, CharWiseParser)
        >>> transcription = (
        ...     get_parser('charwise', 'true')
        ...     .parse_single_variant_transcription('hello!')
        ... )
        >>> [token.value for token in transcription.blocks]
        ['h', 'e', 'l', 'l', 'o']
    """
    ...