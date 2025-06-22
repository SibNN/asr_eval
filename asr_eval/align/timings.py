from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from gigaam.model import GigaAMASR

from asr_eval.align.data import Token, MultiVariant, Anything
from asr_eval.align.parsing import split_text_into_tokens
from asr_eval.ctc.base import ctc_mapping
from asr_eval.ctc.forced_alignment import forced_alignment
from asr_eval.models.gigaam import FREQ, decode, encode, transcribe_with_gigaam_ctc, GigaAMEncodeError

@dataclass
class _TokenEncoded:
    ref: Token
    idxs: list[int] | Literal['not_possible', 'anything']
    
    @classmethod
    def from_token(cls, token: Token, model: GigaAMASR) -> _TokenEncoded:
        if isinstance(token.value, Anything):
            tokens = 'anything'
        else:
            try:
                tokens = encode(model, token.value)
            except GigaAMEncodeError:
                tokens = 'not_possible'
        return _TokenEncoded(ref=token, idxs=tokens)

@dataclass
class _MultiVariantEncoded:
    options: list[list[_TokenEncoded]]  # may be in different order comparing to ref
    ref: MultiVariant

    @classmethod
    def from_multivariant(cls, block: MultiVariant, model: GigaAMASR) -> _MultiVariantEncoded:
        options = [
            [_TokenEncoded.from_token(token, model) for token in option]
            for option in block.options
        ]
        options = sorted(options, key=cls.get_option_value)[::-1]
        return _MultiVariantEncoded(options, block)
    
    @staticmethod
    def get_option_value(option: list[_TokenEncoded]) -> float:
        joined_text = ' '.join(
            token.ref.value
            for token in option
            if not isinstance(token.ref.value, Anything)
        )
        return len(joined_text) + (0.5 if len(option) == 1 else 0)
    
    @staticmethod
    def _is_valid(option: list[_TokenEncoded]) -> bool:
        return len(option) > 0 and all(t.idxs not in ('not_possible', 'anything') for t in option)
    
    def filter_valid_options(self) -> list[list[_TokenEncoded]]:
        return [option for option in self.options if self._is_valid(option)]
    
    def filter_invalid_options(self) -> list[list[_TokenEncoded]]:
        return [option for option in self.options if not self._is_valid(option)]


def fill_word_timings_inplace(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    tokens: list[Token | MultiVariant],
):
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    finish_time = len(waveform) / 16_000
    
    encoded_multivariant = [
        _TokenEncoded.from_token(x, model)
        if isinstance(x, Token)
        else _MultiVariantEncoded.from_multivariant(x, model)
        for x in tokens
    ]

    # select best (longest and valid) option for each multivariant block, also skip Anything tokens
    baseline: list[_TokenEncoded] = []
    for x in encoded_multivariant:
        match x:
            case _TokenEncoded():
                assert x.idxs != 'not_possible', 'cannot encode'
                if x.idxs != 'anything':
                    baseline.append(x)
            case _MultiVariantEncoded():
                for option in x.options:
                    assert all(t.idxs != 'anything' for t in option), 'cannot encode'
                valid_options = x.filter_valid_options()
                assert len(valid_options) > 0, 'cannot encode'
                baseline += valid_options[0]

    # do force alignment on baseline
    baseline_idxs: list[int] = sum([word.idxs for word in baseline], []) # type: ignore
    _idxs, _probs, spans = forced_alignment(
        outputs.log_probs,
        baseline_idxs,
        blank_id=model.decoding.blank_id,
    )
    for word in baseline:
        spans_for_word = spans[:len(word.idxs)]
        spans = spans[len(word.idxs):]
        word.ref.start_time = spans_for_word[0][0] / FREQ
        word.ref.end_time = spans_for_word[-1][1] / FREQ
        # print(word.ref.value, word.ref.start_time, word.ref.end_time)

    # process the remaining multivariant options
    for block in encoded_multivariant:
        if isinstance(block, _MultiVariantEncoded):
            baseline_option = block.filter_valid_options()[0]
            baseline_start = baseline_option[0].ref.start_time
            baseline_end = baseline_option[-1].ref.end_time
            for option in block.filter_valid_options()[1:] + block.filter_invalid_options():
                # skip the first valid option that was in the baseline
                if len(option) == 0:
                    # skip empty options
                    continue
                elif len(baseline_option) == 1:
                    # scenario 1: baseline has one word, other option has one or many words
                    times = np.linspace(baseline_start, baseline_end, endpoint=True, num=len(option) + 1)
                    for t, (start, end) in zip(option, pairwise(times), strict=True):
                        t.ref.start_time = start
                        t.ref.end_time = end
                elif len(option) == 1:
                    # scenario 2: baseline has many words, other option has one word
                    assert len(option) == 1, 'cannot encode'
                    option[0].ref.start_time = baseline_start
                    option[0].ref.end_time = baseline_end
                else:
                    # scenario 3: baseline has 2 words, other option has 2 words
                    # and either first word matches or second word matches
                    assert len(baseline_option) == 2 and len(option) == 2, 'cannot encode'
                    if (
                        baseline_option[0].ref.value == option[0].ref.value
                        or baseline_option[1].ref.value == option[1].ref.value
                    ):
                        option[0].ref.start_time = baseline_option[0].ref.start_time
                        option[0].ref.end_time = baseline_option[0].ref.end_time
                        option[1].ref.start_time = baseline_option[1].ref.start_time
                        option[1].ref.end_time = baseline_option[1].ref.end_time
                # TODO do force alignment for other valid options?

    # process the Anything tokens: expand their time spans as wide as possible
    for i, t in enumerate(encoded_multivariant):
        if isinstance(t, _TokenEncoded) and t.idxs == 'anything':
            prev_end_time = (
                encoded_multivariant[i - 1].ref.end_time
                if i > 0
                else 0
            )
            next_start_time = (
                encoded_multivariant[i + 1].ref.start_time
                if i != len(encoded_multivariant) - 1
                else finish_time
            )
            assert not np.isnan(prev_end_time), 'cannot encode'
            assert not np.isnan(next_start_time), 'cannot encode'
            # TODO make and return deep copy of all refs
            t.ref.start_time = prev_end_time
            t.ref.end_time = next_start_time


def get_word_timings_simple(
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