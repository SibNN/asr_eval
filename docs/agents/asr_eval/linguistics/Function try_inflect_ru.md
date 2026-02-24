# Function try_inflect_ru (defined in asr_eval/linguistics/linguistics.py at lines 101-151)

def try_inflect_ru(
    word: str, original_word: str
) -> tuple[str, typing.Literal['ok', 'ok_manually', 'fail']]:
    """Tries to inflect a Russian lemmatized :code:`word` using
    pymorphy2 to get same form as in :code:`original_word`.

    Useful to restore a word form after correcting misspelled word.
    Returns also a status: 'ok', 'ok_manually', 'fail' (see the code for
    details).

    Examples:
        >>> try_inflect_ru('мемас', 'мэмасы')
        ('мемасы', 'ok')
        >>> try_inflect_ru('антиген', 'онтегенам')
        ('антигенам', 'ok')

    Author: Yana Fitkovskaja; Updated by: Oleg Sedukhin
    """
    ...