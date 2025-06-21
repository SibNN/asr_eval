import re

import numpy as np
import numpy.typing as npt
from gigaam.model import GigaAMASR


from asr_eval.align.data import Token
from asr_eval.align.parsing import split_text_into_tokens
from asr_eval.ctc.base import ctc_mapping
from asr_eval.ctc.forced_alignment import forced_alignment
from asr_eval.models.gigaam import FREQ, decode, encode, transcribe_with_gigaam_ctc


def get_word_timings(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    text: str | None = None,
    normalize: bool = True,
) -> list[Token]:
    '''

    Using GigaAM CTC model, performs either a forced alignment or an argmax prediction, and returns
    a list of words and their timings in seconds:

    [
        (Token('и'), 0.12, 0.16),
        (Token('поэтому'), 0.2, 0.56),
        (Token('использовать'), 0.64, 1.28),
        (Token('их'), 1.32, 1.44),
        (Token('в'), 1.48, 1.56),
        (Token('повседневности'), 1.6, 2.36),
        ...
    ]
    '''
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    if text is None:
        tokens = outputs.log_probs.argmax(axis=1)
    else:
        assert '{' not in text, 'Not implemented for multivariant texts'
        if normalize:
            text = text.lower().replace('ё', 'е').replace('-', ' ')
            for char in ('.', ',', '!', '?', ';', ':', '"', '(', ')', '«', '»', '—'):
                text = text.replace(char, '')
        tokens, _probs, _spans = forced_alignment(
            outputs.log_probs,
            encode(model, text),
            blank_id=model.decoding.blank_id
        )
    letter_per_frame = decode(model, tokens)
    word_timings = [
        Token(
            value=''.join(ctc_mapping(list(match.group()), blank='_')),
            start_time=match.start() / FREQ,
            end_time=match.end() / FREQ,
        )
        for match in re.finditer(r'[а-я]([а-я_]*[а-я])?', letter_per_frame)
    ]

    # fill positions
    if text is not None:
        true_tokens_with_positions = split_text_into_tokens(text)
        for token_to_return, token_with_pos in zip(
            word_timings, true_tokens_with_positions, strict=True
        ):
            assert token_to_return.value == token_with_pos.value
            token_to_return.start_pos = token_with_pos.start_pos
            token_to_return.end_pos = token_with_pos.end_pos

    return word_timings