from typing import Literal

import wordfreq
from pymorphy3 import MorphAnalyzer
from pymorphy3.analyzer import Parse
from pymystem3 import Mystem  # pip install pymystem3


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