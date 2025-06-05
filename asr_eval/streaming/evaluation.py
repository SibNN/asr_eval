import re

from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt

from ..ctc.base import ctc_mapping
from ..ctc.forced_alignment import forced_alignment
from ..models.gigaam import transcribe_with_gigaam_ctc, encode, decode, FREQ


def get_word_timings(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    text: str | None = None,
) -> list[tuple[str, float, float]]:
    '''
    Outputs a list of words and their timings in seconds:

    ([('и', 0.12, 0.16),
        ('поэтому', 0.2, 0.56),
        ('использовать', 0.64, 1.28),
        ('их', 1.32, 1.44),
        ('в', 1.48, 1.56),
        ('повседневности', 1.6, 2.36),
    '''
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    if text is None:
        tokens = outputs.log_probs.argmax(axis=1)
    else:
        tokens, _probs = forced_alignment(
            outputs.log_probs,
            encode(model, text),
            blank_id=model.decoding.blank_id
        )
    letter_per_frame = decode(model, tokens)
    word_timings = [
        (
            ''.join(ctc_mapping(list(match.group()), blank='_')),
            match.start() / FREQ,
            match.end() / FREQ,
        )
        for match in re.finditer(r'[а-я]([а-я_]*[а-я])?', letter_per_frame)
    ]
    return word_timings


def words_count(
    word_timings: list[tuple[str, float, float]],
    time: float,
) -> tuple[int, bool]:
    '''
    Returns a tuple of:
    1. Number of full words in the time span [0, time]
    2. `in_word` flag: is the given time inside a word?
    '''
    count = 0
    in_word = False

    for _word, start, end in word_timings:
        if end <= time:
            count += 1
        else:
            if start < time:
                in_word = True
            break
    
    return count, in_word