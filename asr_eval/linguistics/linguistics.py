from typing import Literal
from itertools import pairwise
import re

import wordfreq
from pymorphy3 import MorphAnalyzer
from pymorphy3.analyzer import Parse
from pymystem3 import Mystem  # pip install pymystem3
import nltk


def word_freq(word: str, lang: str = 'ru') -> float:
    '''
    Get a word frequency in the specified language, according to
    wordfreq.zipf_frequency. Note that wordfreq did not lemmatize words
    before calculating frequency.
    
    If 'word' argument contains several words, frequencies for them are
    combined using the formula 1 / f = 1 / f1 + 1 / f2 + ...
    
    Examples for 'ru':
    word_freq('трофонопсис') == 0
    word_freq('трубочник') == 1.06
    word_freq('трещотка') == 2.05
    word_freq('барсук') == 3.01
    word_freq('железный') == 4.02
    word_freq('девушка') == 5.08
    word_freq('до') == 6.38
    
    See list of available languages in
    `wordfreq.available_languages(wordlist='large')`.
    '''
    return wordfreq.zipf_frequency(word.lower(), lang, wordlist='large')


_mystem: Mystem | None = None


def lemmatize_ru(word: str) -> str:
    '''
    Lemmatizes a Russian word using Mystem. We prefer it over pymorphy2
    due to possibly less frequent errors.
    
    Leaves non-russian words unchanged.
    
    TODO: maybe Mystem would lemmatize better if the whole sentence is passed?
    
    Raises ValueError if Mystem founds zero or more than one word in the 'word' argument.
    '''
    global _mystem
    if _mystem is None:
        _mystem = Mystem()
    result: list[str] = _mystem.lemmatize(word) # type: ignore
    assert len(result) == 2 and result[-1] == '\n', f'Got unusual output from Mystem: {result}'
    return result[0]


_morph: MorphAnalyzer | None = None


def try_inflect_ru(word: str, original_word: str) -> tuple[str, Literal['ok', 'ok_manually', 'fail']]:
    '''
    Tries to inflect a Russian `lemmatized_word` using pymorphy2 to the same
    form as in `original_word`.
    
    Useful to restore a word form after correcting misspelled word.
    
    Returns also a status: 'ok', 'ok_manually', 'fail' (see code for details).
    
    Examples:
    try_inflect_ru('мемас', 'мэмасы') == ('мемасы', 'ok')
    try_inflect_ru('антиген', 'онтегенам') == ('антигенам', 'ok')
    
    Author: Yana Fitkovskaya
    Updated by: Oleg Sedukhin
    '''
    global _morph
    if _morph is None:
        _morph = MorphAnalyzer(path=None)
    parsed_new: Parse = _morph.parse(word.lower())[0] # type: ignore
    parsed_orig: Parse = _morph.parse(original_word.lower())[0] # type: ignore
    
    ## option 1: auto
    
    if (inflected := parsed_new.inflect(parsed_orig.tag.grammemes)): # type: ignore
        return inflected.word, 'ok' # type: ignore
    
    ## option 2: manually
    
    # creating the required tags based on parsed_orig
    needed_grammemes: set[str] = (
        set(parsed_orig.tag.grammemes) - {parsed_orig.tag.POS} # type: ignore
    )
    # removing gender if it conflicts
    if parsed_new.tag.gender != parsed_orig.tag.gender: # type: ignore
        needed_grammemes -= {'masc', 'femn', 'neut'}
    # trying to inflect with the updated tags
    
    print(needed_grammemes)
    if (inflected := parsed_new.inflect(needed_grammemes)): # type: ignore
        return inflected.word, 'ok_manually' # type: ignore
    
    ## option 3: return unchanged
    
    return word, 'fail'




def split_text_into_sentences(
    text: str,
    language: Literal['russian', 'english'] = 'russian',
    max_symbols: int | None = None,
    merge_smaller_than: int | None = None,
) -> list[str]:
    '''
    Split text into sentences using nltk.
    
    If some sentence has more than max_symbols symbols, will split it further
    by space symbols so that each part has no more than `max_symbols` symbols.
    If a single word has more than `max_symbols` symbols, it will be kept as is
    (no truncation or dividing a word into parts).
    
    If merge_smaller_than is specified, will try to merge sentences smaller than
    the specified value, without exceeding max_symbols.
    '''
    nltk.downloader._downloader.download('punkt_tab', quiet=True) # type: ignore
    
    sentences: list[str] = []
    for sentence in nltk.sent_tokenize(text, language=language): # type: ignore
        sentence = sentence.strip()
        if max_symbols is not None and len(sentence) > max_symbols:
            sentences += split_text_by_space(sentence, max_symbols=max_symbols)
        else:
            sentences.append(sentence)
    
    if merge_smaller_than is not None:
        assert max_symbols is not None, (
            'Setting min_symbols is meaningless without setting max_symbols'
        )
        while True:
            sentences, was_merged = _text_iterative_merge_step(
                sentences, merge_smaller_than=merge_smaller_than, max_part_size=max_symbols,
            )
            if not was_merged:
                break
    
    return sentences


def split_text_by_space(text: str, max_symbols: int) -> list[str]:
    '''
    Split text into parts by space symbols so that each part has no more
    than `max_symbols` symbols. If a single word has more than `max_symbols`
    symbols, it will be kept as is (no truncation or dividing a word into parts).
    '''
    parts: list[list[re.Match[str]]] = []
    for word_match in re.finditer(r'[^\s]+', text):
        if len(parts) > 0 and word_match.end() - parts[-1][0].start() < max_symbols:
            # add to the existing part
            parts[-1].append(word_match)
        else:
            # start a new part
            parts.append([word_match])
    
    return [
        text[part[0].start() : part[-1].end()]
        for part in parts
    ]
    
    
def _text_iterative_merge_step(
    parts: list[str], max_part_size: int, merge_smaller_than: int | None = None,
) -> tuple[list[str], bool]:
    if len(parts) < 2:
        return parts, False
    
    # preparing candidate splits
    candidates_and_scores: list[tuple[int, float]] = []
    for i, (part1, part2) in enumerate(pairwise(parts)):
        merged_len = len(part1) + len(part2) + 1
        if merged_len <= max_part_size and (
            merge_smaller_than is None
            or len(part1) < merge_smaller_than
            or len(part2) < merge_smaller_than
        ):
            candidates_and_scores.append((i, merged_len))
    
    if not candidates_and_scores:
        return parts, False
    
    # find two consecutive segments with the smallest total length
    best_i, _ = min(candidates_and_scores, key=lambda x: x[1])
    
    head = parts[:best_i]
    part1 = parts[best_i]
    part2 = parts[best_i + 1]
    tail = parts[best_i + 2:]

    return head + [part1 + ' ' + part2] + tail, True