# Function lemmatize_ru (defined in asr_eval/linguistics/linguistics.py at lines 46-70)

def lemmatize_ru(word: str) -> str:
    """Lemmatizes a Russian word using Mystem. We prefer it over
    pymorphy2 due to possibly less frequent errors.

    Leaves non-Russian words unchanged.

    TODO: maybe Mystem would lemmatize better if the whole sentence is
    passed?

    Raises:
        ValueError: If Mystem founds zero or more than one word in the
            :code:`word` argument.
    """
    ...