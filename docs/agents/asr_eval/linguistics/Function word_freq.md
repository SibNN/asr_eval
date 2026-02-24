# Function word_freq (defined in asr_eval/linguistics/linguistics.py at lines 13-41)

def word_freq(word: str, lang: str = 'ru') -> float:
    """Get a word frequency for the specified language, according to
    :code:`wordfreq.zipf_frequency`. Note that wordfreq does not
    lemmatize words before calculating frequency.

    If :code:`word` argument contains several words, frequencies for
    them are combined using the formula 1 / f = 1 / f1 + 1 / f2 + ...
    (a default behaviour in wordfreq).

    Examples for 'ru':

    .. code-block:: python

        word_freq('трофонопсис') == 0
        word_freq('трубочник') == 1.06
        word_freq('трещотка') == 2.05
        word_freq('барсук') == 3.01
        word_freq('железный') == 4.02
        word_freq('девушка') == 5.08
        word_freq('до') == 6.38

    See list of available languages in
    :code:`wordfreq.available_languages(wordlist='large')`.
    """
    ...