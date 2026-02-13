from __future__ import annotations
from typing import Literal

from asr_eval.align.parsing import Parser


parsers_registry: dict[tuple[str, Literal['true', 'pred']], type[Parser]] = {}
"""A registry where the registered parsers are stored."""

parser_instances: dict[tuple[str, Literal['true', 'pred']], Parser] = {}


def register_parsers(
    name: str, true_parser: type[Parser], pred_parser: type[Parser]
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
    assert (name, 'true') not in parsers_registry
    parsers_registry[name, 'true'] = true_parser
    
    assert (name, 'pred') not in parsers_registry
    parsers_registry[name, 'pred'] = pred_parser


def get_parser(name: str, type: Literal['true', 'pred']):
    """Retrieve a registered parser for annotation (:code:`type='true'`)
    of prediction (:code:`type='pred'`). Will instantiate this parser
    and return it on all subsequent calls (useful for parsers containing
    neural text normalizers).
    """
    assert (name, type) in parsers_registry, f'Parser {name} is not registered'
    if (name, type) in parser_instances:
        return parser_instances[name, type]
    parser_instances[name, type] = parser = parsers_registry[name, type]()
    return parser


def unload_parser(name: str):
    assert (name, type) in parsers_registry, f'Parser {name} is not registered'
    parser_instances.pop((name, 'true'), None)
    parser_instances.pop((name, 'pred'), None)