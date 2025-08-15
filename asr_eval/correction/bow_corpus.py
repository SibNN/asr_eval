import re
from typing import Literal

from tqdm.auto import tqdm

from asr_eval.linguistics.linguistics import word_freq, lemmatize_ru


__all__ = [
    'prepare_domain_specific_bag_of_words_corpus',
]


def prepare_domain_specific_bag_of_words_corpus(
    corpus: str,
    pattern: str = r'\w+',
    lemmatize: Literal['add', 'replace', 'no'] = 'add',
    wordfreq_threshold: float | None = 2,
    wordfreq_lang: str = 'ru',
    pbar: bool = False,
) -> set[str]:
    '''
    Extracts words from domain specific corpus or dictionary. Extracts words, for each
    word adds/replaces with lemmatized form. Filters out too frequent words based on
    wordfreq_threshold.
    '''
    words: set[str] = set(re.findall(pattern, corpus.lower()))
    
    words |= {word.replace('ё', 'е') for word in words if 'ё' in word}
    
    if lemmatize != 'no':
        lemmatized_words = {
            lemmatize_ru(word) for word in tqdm(words, disable=not pbar, desc='lematizing')
        }
        match lemmatize:
            case 'add':
                words |= lemmatized_words
            case 'replace':
                words = lemmatized_words
    
    if wordfreq_threshold is not None:
        words = {
            word for word in tqdm(words, disable=not pbar, desc='wordfreq')
            if word_freq(word, lang=wordfreq_lang) < wordfreq_threshold
        }
    
    return words