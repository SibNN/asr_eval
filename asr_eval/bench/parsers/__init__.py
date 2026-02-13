from __future__ import annotations
from asr_eval.bench.parsers._registry import (
    parsers_registry,
    parser_instances,
    register_parsers,
    get_parser,
    unload_parser,
)
from asr_eval import CACHE_DIR
from asr_eval.align.parsing import Parser
from asr_eval.normalizing.filler_words import RuFillerRemover
from asr_eval.normalizing.silero import RuSileroNormalizer
from asr_eval.normalizing.translit import TranslitNormalizer
from asr_eval.utils.cacheable import DiskCacheable


__all__ = [
    'register_parsers',
    'parsers_registry',
    'get_parser',
    'unload_parser',
    'parser_instances',
    'RuNormParser',
]


register_parsers('default', true_parser=Parser, pred_parser=Parser)
    

class RuNormParser(Parser):
    """A parser with Russian text normalization. Includes translit
    normalization, Silero normalization and filler words removal."""
    def __init__(self):
        filler_remover = RuFillerRemover()
        translit_normalizer = TranslitNormalizer()
        silero_normalizer = DiskCacheable(
            RuSileroNormalizer(),
            cache_path=CACHE_DIR / 'sliero_normalizer_cache.db',
        )

        def normalize(text: str) -> str:
            text = translit_normalizer(text)
            text = silero_normalizer(text)
            text = filler_remover(text)
            return text
        
        super().__init__(preprocessing=normalize)


register_parsers('ru-norm', true_parser=RuNormParser, pred_parser=RuNormParser)
    

class RuNormLiteParser(Parser):
    def __init__(self):
        filler_remover = RuFillerRemover()
        translit_normalizer = TranslitNormalizer()

        def normalize(text: str) -> str:
            text = translit_normalizer(text)
            text = filler_remover(text)
            return text
        
        super().__init__(preprocessing=normalize)


register_parsers('ru-norm-lite', true_parser=RuNormLiteParser, pred_parser=RuNormLiteParser)