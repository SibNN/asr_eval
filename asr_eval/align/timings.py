from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from itertools import pairwise

import numpy as np
from gigaam.model import GigaAMASR

from ..utils.types import FLOATS
from .data import Token, MultiVariant, Anything
from .parsing import parse_single_variant_string
from ..ctc.base import ctc_mapping
from ..ctc.forced_alignment import forced_alignment
from ..models.base.longform import LongformCTC
from ..models.gigaam_wrapper import FREQ, GigaAMShortformCTC, gigaam_decode, gigaam_encode, GigaAMEncodeError
from ..utils.misc import self_product_nonequal


__all__ = [
    'CannotFillTimings',
    'fill_word_timings_inplace',
    'get_word_timings_simple',
]


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
                tokens = gigaam_encode(model, token.value)
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


class CannotFillTimings(ValueError):
    '''
    Indicates that we failed to fill .start_time and .end_time for words.
    '''
    pass
    

def _propagate_timings(
    from_block: list[_TokenEncoded],
    to_block: list[_TokenEncoded],
) -> bool:
    from_block = from_block.copy()
    to_block = to_block.copy()
    
    shared_head_from: list[_TokenEncoded] = []
    shared_head_to: list[_TokenEncoded] = []
    shared_tail_from: list[_TokenEncoded] = []
    shared_tail_to: list[_TokenEncoded] = []
    
    # cut an equal head
    while len(from_block) and len(to_block) and from_block[0].ref.value == to_block[0].ref.value:
        shared_head_from.append(from_block[0])
        shared_head_to.append(to_block[0])
        from_block = from_block[1:]
        to_block = to_block[1:]
        
    # cut an equal tail
    while len(from_block) and len(to_block) and from_block[-1].ref.value == to_block[-1].ref.value:
        shared_tail_from.insert(0, from_block[-1])
        shared_tail_to.insert(0, to_block[-1])
        from_block = from_block[:-1]
        to_block = to_block[:-1]
    
    def propagate_pairwise(_from: list[_TokenEncoded], _to: list[_TokenEncoded]):
        for token1, token2 in zip(_from, _to, strict=True):
            token2.ref.start_time = token1.ref.start_time
            token2.ref.end_time = token1.ref.end_time
    
    # study the different part
    if len(from_block) <= 1 or len(to_block) <= 1:
        propagate_pairwise(shared_head_from, shared_head_to)
        propagate_pairwise(shared_tail_from, shared_tail_to)
        # propagate timings for different part
        if len(from_block) == 0:
            return False
            # for token in to_block:
            #     token.ref.start_time = np.inf  # use as a temporary placeholder
            #     token.ref.end_time = np.inf  # use as a temporary placeholder
        elif len(from_block) == 1:
            times = np.linspace(
                from_block[0].ref.start_time,
                from_block[0].ref.end_time,
                num=len(to_block) + 1
            )
            for (time1, time2), token in zip(pairwise(times), to_block):
                token.ref.start_time = time1
                token.ref.end_time = time2
        elif len(to_block) == 1:
            to_block[0].ref.start_time = from_block[0].ref.start_time
            to_block[0].ref.end_time = from_block[-1].ref.end_time
        return True

    # custom rules
    for currency_symbol in '$', '€', '¥', '£', '₽', '₹', '¥':
        # for example, from ['100', '000', '$'] to from ['$', '100', '000']
        if (
            len(from_block) == len(to_block) > 1
            and from_block[-1].ref.value == currency_symbol
            and to_block[0].ref.value == currency_symbol
        ):
            propagate_pairwise(from_block[:-1], to_block[1:])
            to_block[0].ref.start_time = to_block[1].ref.start_time
            to_block[0].ref.end_time = to_block[1].ref.end_time
    
    return False


def _print_propagation(
    from_block: list[_TokenEncoded],
    to_block: list[_TokenEncoded],
):
    str1 = ', '.join(
        f'{t.ref.value} ({t.ref.start_time:.1f}-{t.ref.end_time:.1f})'
        for t in from_block
    )
    str2 = ', '.join(
        f'{t.ref.value} ({t.ref.start_time:.1f}-{t.ref.end_time:.1f})'
        for t in to_block
    )
    print(f'Propagated timings from [{str1}] to [{str2}]')
    

def fill_word_timings_inplace(
    model: GigaAMShortformCTC,
    waveform: FLOATS,
    tokens: list[Token | MultiVariant],
    verbose: bool = False,
):
    '''
    Given a Russian tokenized multivariant or single-variant transcription and the corresponding
    audio, fills .start_time and .end_time for each Token using GigaAM CTC model.
    
    In each multivariant block, tries to find fully-Russian option and uses it as a baseline,
    then tries to fill timings for other (possibly not Russian) options based on the baseline
    option timings.
    
    Raises CannotFillTimings if this turns out to be impossible.
    '''
    log_probs = LongformCTC(model).ctc_log_probs([waveform])[0]
    finish_time = len(waveform) / 16_000
    
    encoded_multivariant = [
        _TokenEncoded.from_token(x, model.model)
        if isinstance(x, Token)
        else _MultiVariantEncoded.from_multivariant(x, model.model)
        for x in tokens
    ]

    # select best (longest and valid) option for each multivariant block, also skip Anything tokens
    baseline: list[_TokenEncoded] = []
    for x in encoded_multivariant:
        match x:
            case _TokenEncoded():
                if x.idxs == 'not_possible':
                    raise CannotFillTimings(f'Cannot encode a single-variant token {x.ref.value}')
                if x.idxs != 'anything':
                    baseline.append(x)
            case _MultiVariantEncoded():
                for option in x.options:
                    if any(t.idxs == 'anything' for t in option):
                        raise CannotFillTimings('Cannot process Anything() in a multivatiant block')
                valid_options = x.filter_valid_options()
                if len(valid_options) == 0:
                    raise CannotFillTimings(
                        'Cannot encode any option in a multivariant block'
                        + str([[t.value for t in option] for option in x.ref.options])
                    )
                baseline += valid_options[0]

    # do force alignment on baseline
    baseline_idxs: list[int] = sum([word.idxs for word in baseline], []) # type: ignore
    _idxs, _probs, spans = forced_alignment(
        log_probs,
        baseline_idxs,
        blank_id=model.model.decoding.blank_id,
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
            # propagate timings: first propagate for equal length (stage 1), then for others
            for stage in (1, 2):
                while True:
                    for option1, option2 in self_product_nonequal(block.options, triangle=False):
                        if (
                            len(option1)
                            and all(t.ref.is_timed for t in option1)
                            and not all(t.ref.is_timed for t in option2)
                            and (stage != 1 or len(option1) == len(option2))
                        ):
                            can_propagate = _propagate_timings(from_block=option1, to_block=option2)
                            if can_propagate:
                                if verbose:
                                    _print_propagation(from_block=option1, to_block=option2)
                                break
                    else:
                        break
            
            # check if all options are now timed
            for option in block.options:
                if not all(t.ref.is_timed for t in option):
                    raise CannotFillTimings('Cannot fill for', [t.ref.value for t in option])

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
            if np.isnan(prev_end_time):
                    raise CannotFillTimings('Cannot find a left boundary for Anything()')
            if np.isnan(next_start_time):
                    raise CannotFillTimings('Cannot find a right boundary for Anything()')
            # TODO make and return deep copy of all refs
            t.ref.start_time = prev_end_time
            t.ref.end_time = next_start_time


def get_word_timings_simple(
    model: GigaAMShortformCTC,
    waveform: FLOATS,
    text: str | None = None,
    normalize: bool = True,
) -> list[Token]:
    '''
    A simplified version of `fill_word_timings_inplace`, accepts raw text as string, instead of
    parsed text. Does not support multivariance.

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
    
    If text is not provided, uses GigaAM argmax predictions instead.
    
    Raises GigaAMEncodeError if cannot encode some letters in the text.
    '''
    log_probs = LongformCTC(model).ctc_log_probs([waveform])[0]
    if text is None:
        tokens = log_probs.argmax(axis=1)
    else:
        assert '{' not in text, 'Not implemented for multivariant texts'
        if normalize:
            text = text.lower().replace('ё', 'е').replace('-', ' ')
            for char in ('.', ',', '!', '?', ';', ':', '"', '(', ')', '«', '»', '—'):
                text = text.replace(char, '')
        tokens, _probs, _spans = forced_alignment(
            log_probs,
            gigaam_encode(model.model, text),
            blank_id=model.model.decoding.blank_id
        )
    letter_per_frame = gigaam_decode(model.model, tokens)
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
        true_tokens_with_positions = parse_single_variant_string(text)
        for token_to_return, token_with_pos in zip(
            word_timings, true_tokens_with_positions, strict=True
        ):
            assert token_to_return.value == token_with_pos.value
            token_to_return.start_pos = token_with_pos.start_pos
            token_to_return.end_pos = token_with_pos.end_pos

    return word_timings