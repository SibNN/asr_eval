from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from itertools import pairwise

import numpy as np

from asr_eval.models.base.interfaces import CTC
from asr_eval.utils.types import FLOATS
from asr_eval.align.transcription import (
    Transcription,
    Token,
    MultiVariantBlock,
    Wildcard,
)
from asr_eval.ctc.forced_alignment import forced_alignment
from asr_eval.models.base.longform import LongformCTC


__all__ = [
    'fill_word_timings_inplace',
    'CannotFillTimings',
]


@dataclass
class _TokenEncoded:
    """A wrapper for `Token` that keeps character indices in model's
    vocab.
    """
    ref: Token
    idxs: list[int] | Literal['not_possible', 'wildcard']
    
    @classmethod
    def from_token(cls, token: Token, vocab: Sequence[str]) -> _TokenEncoded:
        if isinstance(token.value, Wildcard):
            tokens = 'wildcard'
        else:
            try:
                tokens = [vocab.index(c) for c in token.value]
            except ValueError:
                tokens = 'not_possible'
        return _TokenEncoded(ref=token, idxs=tokens)

@dataclass
class _MultiVariantEncoded:
    """A wrapper for `MultiVariantBlock` that keeps character indices in
    model's vocab.
    """
    # options may be in different order comparing to ref
    options: list[list[_TokenEncoded]]
    ref: MultiVariantBlock

    @classmethod
    def from_multivariant(
        cls, block: MultiVariantBlock, vocab: Sequence[str]
    ) -> _MultiVariantEncoded:
        options = [
            [_TokenEncoded.from_token(token, vocab) for token in option]
            for option in block.options
        ]
        options = sorted(options, key=cls.get_option_value)[::-1]
        return _MultiVariantEncoded(options, block)
    
    @staticmethod
    def get_option_value(option: list[_TokenEncoded]) -> float:
        joined_text = ' '.join(
            token.ref.value
            for token in option
            if not isinstance(token.ref.value, Wildcard)
        )
        return len(joined_text) + (0.5 if len(option) == 1 else 0)
    
    @staticmethod
    def _is_valid(option: list[_TokenEncoded]) -> bool:
        return (
            len(option) > 0
            and all(t.idxs not in ('not_possible', 'wildcard') for t in option)
        )
    
    def filter_valid_options(self) -> list[list[_TokenEncoded]]:
        return [
            option for option in self.options if self._is_valid(option)
        ]
    
    def filter_invalid_options(self) -> list[list[_TokenEncoded]]:
        return [
            option for option in self.options if not self._is_valid(option)
        ]


class CannotFillTimings(ValueError):
    """ An exception raised from
    :func:`~asr_eval.align.timings.fill_word_timings_inplace` that
    indicates a failure to fill timings, usually because of absence of
    the required characters in the model vocab.
    """
    pass
    

def _propagate_timings(
    from_block: list[_TokenEncoded],
    to_block: list[_TokenEncoded],
) -> bool:
    """Tries to propagate timings from one option in `MultiVariantBlock`
    to another.
    """
    from_block = from_block.copy()
    to_block = to_block.copy()
    
    shared_head_from: list[_TokenEncoded] = []
    shared_head_to: list[_TokenEncoded] = []
    shared_tail_from: list[_TokenEncoded] = []
    shared_tail_to: list[_TokenEncoded] = []
    
    # cut an equal head
    while (
        len(from_block)
        and len(to_block)
        and from_block[0].ref.value == to_block[0].ref.value
    ):
        shared_head_from.append(from_block[0])
        shared_head_to.append(to_block[0])
        from_block = from_block[1:]
        to_block = to_block[1:]
        
    # cut an equal tail
    while (
        len(from_block)
        and len(to_block)
        and from_block[-1].ref.value == to_block[-1].ref.value
    ):
        shared_tail_from.insert(0, from_block[-1])
        shared_tail_to.insert(0, to_block[-1])
        from_block = from_block[:-1]
        to_block = to_block[:-1]
    
    def propagate_pairwise(
        _from: list[_TokenEncoded], _to: list[_TokenEncoded]
    ):
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
    model: CTC,
    waveform: FLOATS,
    transcription: Transcription,
    verbose: bool = False,
):
    """Fills :attr:`~asr_eval.align.transcription.Token.start_time` and
    :attr:`~asr_eval.align.transcription.Token.end_time` in
    transcription via forced alignment.
    
    Args:
        model: A model with
            :class:`~asr_eval.models.base.interfaces.CTC` interface.
            Normally it should support all the characters found in
            `transcription` ignoring case. However, if it does not
            support some options in a multivariant block, the function
            will run some propagation rules and try to fill the timings.
            For example, if there is no digits in the model's vocab,
            it will still able to fill the timings for a block
            :code:`{1|one}`. We first fill the timings for the word
            "one", then mirror them to the word "1".
        waveform: The audio 16000 kHz, possibly long. For long audios
            will wrap the CTC model into a
            :class:`~asr_eval.models.base.longform.LongformCTC` that
            works via logit averaging of uniform chunks with overlap.
        transcription: A single-variant or multivariant transcription.
        verbose: Enable debug output.
    
    Raises:
        CannotFillTimings: if cannot fill timings due to the absence of
            the required characters in the model's vocab, or other
            limitations of the algorithm.
    
    Example:
        >>> from typing import cast
        >>> from datasets import load_dataset, Audio
        >>> from asr_eval.align.timings import fill_word_timings_inplace
        >>> from asr_eval.bench.datasets import get_dataset
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.plots import draw_timed_transcription
        >>> dataset = (
        ...     load_dataset('PolyAI/minds14', name='en-GB', split='train')
        ...     .cast_column('audio', Audio(16_000))
        ... )
        >>> sample = dataset[0]
        >>> transcription = DEFAULT_PARSER \\
        ...     .parse_transcription(sample['transcription'])
        >>> waveform = sample['audio']['array']
        
        >>> # # to display the audio:
        >>> # import IPython.display    # doctest: +SKIP
        >>> # IPython.display.Audio(waveform, rate=16_000)  # doctest: +SKIP
        
        >>> # For English:
        >>> from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper
        >>> model = Wav2vec2Wrapper('facebook/wav2vec2-base-960h')
        
        >>> # # For Russian:
        >>> # from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
        >>> # model = GigaAMShortformCTC('v2')
        
        >>> fill_word_timings_inplace(model, waveform, transcription)
        >>> print(transcription.blocks[:6]) # doctest: +NORMALIZE_WHITESPACE
        (Token(i, t=(1.0, 1.0)), Token(want, t=(1.1, 1.3)), Token(to, t=(1.3, 1.5)),
         Token(pay, t=(2.0, 2.2)), Token(a, t=(2.3, 2.4)), Token(bill, t=(2.4, 2.7)))
         
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(8, 2)) # doctest: +ELLIPSIS
        <...>
        >>> draw_timed_transcription(transcription, y_tick_width=0.02) # doctest: +ELLIPSIS
        >>> plt.plot(np.arange(len(waveform)) / 16000, waveform, alpha=0.3) # doctest: +ELLIPSIS
        [...]
    
    .. image:: images/docstrings/fill_word_timings_inplace.png

    """
    if not isinstance(model, LongformCTC):
        model = LongformCTC(model)
    log_probs = model.ctc_log_probs([waveform])[0]
    finish_time = len(waveform) / 16_000
    
    lowercase_vocab = [c.lower() for c in model.vocab]
    
    encoded_multivariant = [
        _TokenEncoded.from_token(x, lowercase_vocab)
        if isinstance(x, Token)
        else _MultiVariantEncoded.from_multivariant(x, lowercase_vocab)
        for x in transcription.blocks
    ]

    # select best (longest and valid) option for each multivariant block, also
    # skip Wildcard tokens
    baseline: list[_TokenEncoded] = []
    for x in encoded_multivariant:
        match x:
            case _TokenEncoded():
                if x.idxs == 'not_possible':
                    raise CannotFillTimings(
                        f'Cannot encode token {x.ref.value}'
                        f'\nOne of the characters is not in the model vocab.'
                    )
                if x.idxs != 'wildcard':
                    baseline.append(x)
            case _MultiVariantEncoded():
                for option in x.options:
                    if any(t.idxs == 'wildcard' for t in option):
                        raise CannotFillTimings(
                            'Cannot process Wildcard() in a multivatiant block'
                        )
                valid_options = x.filter_valid_options()
                if len(valid_options) == 0:
                    raise CannotFillTimings(
                        'Cannot encode any option in a multivariant block'
                        + str([
                            [t.value for t in option]
                            for option in x.ref.options
                        ])
                    )
                baseline += valid_options[0]

    # do force alignment on baseline
    baseline_idxs: list[int] = sum([word.idxs for word in baseline], []) # type: ignore
    _idxs, _probs, spans = forced_alignment(
        log_probs, baseline_idxs, blank_id=model.blank_id,
    )
    for word in baseline:
        spans_for_word = spans[:len(word.idxs)]
        spans = spans[len(word.idxs):]
        word.ref.start_time = spans_for_word[0][0] * model.tick_size
        word.ref.end_time = spans_for_word[-1][1] * model.tick_size
        # print(word.ref.value, word.ref.start_time, word.ref.end_time)

    # process the remaining multivariant options
    for block in encoded_multivariant:
        if isinstance(block, _MultiVariantEncoded):
            # propagate timings: first propagate for equal length (stage 1),
            # then for others
            for stage in (1, 2):
                while True:
                    change_done = False
                    for option1 in block.options:
                        for option2 in block.options:
                            if option1 is option2:
                                continue
                            if (
                                len(option1)
                                and all(t.ref.is_timed for t in option1)
                                and not all(t.ref.is_timed for t in option2)
                                and (
                                    stage != 1
                                    or len(option1) == len(option2)
                                )
                            ):
                                can_propagate = _propagate_timings(
                                    from_block=option1, to_block=option2
                                )
                                if can_propagate:
                                    change_done = True
                                    if verbose:
                                        _print_propagation(
                                            from_block=option1,
                                            to_block=option2,
                                        )
                                    break
                    if not change_done:
                        break
            
            # check if all options are now timed
            for option in block.options:
                if not all(t.ref.is_timed for t in option):
                    block_text = transcription.text[block.ref.start_pos \
                        :block.ref.end_pos]
                    raise CannotFillTimings(
                        'Cannot fill timings for'
                        f' {[t.ref.value for t in option]} in {block_text}'
                    )
                

    # process the Wildcard tokens: expand their time spans as wide as possible
    for i, t in enumerate(encoded_multivariant):
        if isinstance(t, _TokenEncoded) and t.idxs == 'wildcard':
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
                    raise CannotFillTimings(
                        'Cannot find a left boundary for Wildcard()'
                    )
            if np.isnan(next_start_time):
                    raise CannotFillTimings(
                        'Cannot find a right boundary for Wildcard()'
                    )
            # TODO make and return deep copy of all refs
            t.ref.start_time = prev_end_time
            t.ref.end_time = next_start_time
        
    for token in transcription.list_all_tokens():
        assert not np.isnan(token.start_time)
        assert not np.isnan(token.end_time)
        # np.float64 -> float conversion
        token.start_time = float(token.start_time)
        token.end_time = float(token.end_time)