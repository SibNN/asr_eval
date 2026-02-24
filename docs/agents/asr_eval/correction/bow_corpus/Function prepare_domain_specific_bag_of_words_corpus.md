# Function prepare_domain_specific_bag_of_words_corpus (defined in asr_eval/correction/bow_corpus.py at lines 14-49)

def prepare_domain_specific_bag_of_words_corpus(
    corpus: str,
    pattern: str = r'\w+',
    lemmatize: typing.Literal['add', 'replace', 'no'] = 'add',
    wordfreq_threshold: float | None = 2,
    wordfreq_lang: str = 'ru',
    pbar: bool = False,
) -> set[str]:
    """ Extracts words from domain specific corpus or dictionary.

    For each word adds/replaces with lemmatized form. Filters out too
    frequent words based on wordfreq_threshold.
    """
    ...