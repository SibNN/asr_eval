"""Linguistic utilities."""

from asr_eval.linguistics.linguistics import (
    word_freq,
    lemmatize_ru,
    try_inflect_ru,
    split_text_into_sentences,
    split_text_by_space,
)


__all__ = [
    'word_freq',
    'lemmatize_ru',
    'try_inflect_ru',
    'split_text_into_sentences',
    'split_text_by_space',
]