from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from asr_eval.utils.misc import new_uid
from asr_eval.utils.formatting import (
    Formatting, FormattingSpan, apply_formatting
)


__all__ = [
    'Transcription',
    'SingleVariantTranscription',
    'Token',
    'Wildcard',
    'MultiVariantBlock',
    'TranscriptionPath',
    'OuterLoc',
    'InnerLoc',
    'get_outer_slots',
    'get_outer_slots_values',
]


TOKEN_UID = str


@dataclass(slots=True)
class Wildcard:
    """
    Represents a Wilcard symbol <*> in a transcription that matches
    every word sequence, possibly empty.
    """
    
    _SYMBOL = '<*>'
    
    def __eq__(self, other: Any) -> bool:
        return True


@dataclass(slots=True)
class Token:
    """Represents typically a single word or a
    :class:`~asr_eval.align.transcription.Wildcard` symbol in a
    transcription.
    
    Note:
        Typically you don't need to create Token manually. They are created
        automatically when parsing a text with
        :class:`~asr_eval.align.parsing.Parser`.
    
    Token is what the alignment algorithm considers an atom. Two tokens
    may be equal or not. If not equal, this contributes to the error
    count. By default :class:`~asr_eval.align.parsing.Parser` splits a
    text into words, and in this case each token is a words, and we
    obtain WER (word error rate) by aligning them. However, with
    modified :attr:`~asr_eval.align.parsing.Parser.tokenizing` regexp
    a parser may split into characters - then each token stores a single
    character. See the :class:`~asr_eval.align.parsing.Parser` docs for
    defails. There is also a special token with :code:`.value=Wilcard()`
    that matches every token equence, possibly empty.
    """
    
    value: str | Wildcard
    """A text (usually a word) after all the normalization steps, or
    a :class:`~asr_eval.align.transcription.Wildcard` symbol. If not
    wilcard, two tokens match if their texts are equal as strings.
    """
    
    uid: TOKEN_UID = field(default_factory=new_uid)
    """An ID that is unique in the text. Is used to refer the token."""
    
    start_pos: int = 0
    """The start position in the original text (the :code:`.text` field
    of the :class:`~asr_eval.align.transcription.Transcription` the
    token belongs to).
    """
    
    end_pos: int = 0
    """The end position in the original text (the :code:`.text` field
    of the :class:`~asr_eval.align.transcription.Transcription` the
    token belongs to), not inclusive.
    """
    
    start_time: float = np.nan
    """The start time in seconds, is NaN by default, can be filled by
    :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
    """
    
    end_time: float = np.nan
    """The end time in seconds, is NaN by default, can be filled by
    :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
    """

    attrs: dict[str, Any] = field(default_factory=dict[str, Any])
    """Key-value attributes, most often empty.
    
    May be used, for example, for word weights.
    """

    flags: set[str] = field(default_factory=set[str])
    """Flags, most often empty.
    
    May be used to flag specific words to evaluate on.
    """
    
    
    def __repr__(self) -> str:
        reprs = [
            str(self.value),
            # f'pos=({self.start_pos}, {self.end_pos})'
        ]
        if self.is_timed:
            reprs.append(f't=({self.start_time:.1f}, {self.end_time:.1f})')
        if self.attrs:
            reprs.append(
                'attrs:'
                + ','.join(f'{k}={v}' for k, v in self.attrs.items())
            )
        if self.flags:
            reprs.append(
                'flags:'
                + ','.join(self.flags)
            )
        return f'Token(' + ', '.join(reprs) + ')'

    @property
    def is_timed(self) -> bool:
        """Are :attr:`~asr_eval.align.transcription.Token.start_time`
        and :attr:`~asr_eval.align.transcription.Token.end_time` not
        NaN?
        """
        return not np.isnan(self.start_time) and not np.isnan(self.end_time)
    
    def to_text(self) -> str:
        """Returns :code:`.value`, or :code:`<*>` if Wildcard, for
        displaying purposes.
        """
        return (
            Wildcard._SYMBOL # pyright: ignore[reportPrivateUsage]
            if isinstance(self.value, Wildcard)
            else self.value
        )


@dataclass(slots=True)
class MultiVariantBlock:
    """
    A multivariant block in a transcription. Contains two or more
    options, each option contains zero or more tokens.
    """
    
    options: list[list[Token]]
    
    start_pos: int = 0
    """Start of the span in the original text, including braces {}."""
    
    end_pos: int = 0
    """End of the span the original text, including braces {}."""
    
    uid: TOKEN_UID = field(default_factory=new_uid)
    """An ID that is unique in the text. Is used to refer the block."""
    
    def __repr__(self) -> str:
        return f'MultiVariantBlock({str(self.options)[1:-1]})'

    @property
    def is_timed(self) -> bool:
        """True if :attr:`~asr_eval.align.transcription.Token.is_timed`
        is true for all tokens in the block.
        """
        return all(all(t.is_timed for t in option) for option in self.options)
    
    @property
    def start_time(self) -> float:
        """
        The earliest
        :attr:`~asr_eval.align.transcription.Token.start_time` across
        all options, or NaN if tokens are not timed.
        """
        start_times = [
            option[0].start_time for option in self.options if len(option)
        ]
        assert len(start_times), (
            'All options are empty in a MultiVariant block, should not happen'
        )
        # `min` builtin works incorrectly: min(-1.0, np.nan) --> -1.0
        return np.min(start_times)
    
    @property
    def end_time(self) -> float:
        """
        The latest :attr:`~asr_eval.align.transcription.Token.end_time`
        across all options, or NaN if tokens are not timed.
        """
        end_times = [
            option[-1].end_time for option in self.options if len(option)
        ]
        assert len(end_times), (
            'All options are empty in a MultiVariant block, should not happen'
        )
        return np.max(end_times)
    
    def get_option_text(self, option_index: int) -> str:
        """For displaying purposes."""
        return ' '.join(
            t.to_text() for t in self.options[option_index]
        )

    def to_text(self) -> str:
        """For displaying purposes."""
        return '{' + '|'.join([
            self.get_option_text(i) for i in range(len(self.options))
        ]) + '}'
    

@dataclass(frozen=True)
class Transcription:
    """
    A transcription that is normalized and parsed into words.
    
    Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_transcription`.
    May contain zero or more multivariant blocks or
    :class:`~asr_eval.align.transcription.Wildcard` insertions. Stores
    `.text` field (the original text) and list of tokens (words) that
    keep references to the positions in the original text.
    A generic for two transcription subclasses:

    Has a subclass
    :class:`~asr_eval.align.transcription.SingleVariantTranscription`
    used for typing clarity where we do not expect multivariance (mainly
    for predictions). Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_single_variant_transcription`.
    
    Example:
    
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> transcription = DEFAULT_PARSER.parse_transcription(
        ...     "Alexa skip to friday {skip to friday} "
        ...     ", {Don't|Do not} need another sad day <*>"
        ... )
        >>> print(transcription.blocks) # doctest: +NORMALIZE_WHITESPACE
        (Token(alexa),
         Token(skip),
         Token(to),
         Token(friday),
         MultiVariantBlock([Token(skip), Token(to), Token(friday)], []),
         MultiVariantBlock([Token(don), Token(t)], [Token(do), Token(not)]),
         Token(need),
         Token(another),
         Token(sad),
         Token(day),
         Token(Wildcard()))
         
         >>> from IPython.display import HTML  # doctest: +SKIP
         >>> HTML(transcription.colorize(color_mode='html')) # doctest: +SKIP
         
        .. raw:: html
        
            <iframe src="_static/transcription_docstring.html"
                style="border: none; width: 100%; height: 50px; overflow: hidden;"></iframe>
    
    Transcription tokens can have attributes and flags, see examples in
    the :class:`~asr_eval.align.parsing.Parser` docs, see also
    fields :attr:`~asr_eval.align.transcription.Token.attrs` and
    :attr:`~asr_eval.align.transcription.Token.flags`.
    """
    
    text: str
    """The original text that was parsed into words."""
    
    blocks: tuple[Token | MultiVariantBlock, ...]
    """For single-varian transcription is a list of
    :class:`~asr_eval.align.transcription.Token`. For multivariant
    transcription may also contain zero or more
    :class:`~asr_eval.align.transcription.MultiVariantBlock`.
    """
    
    def list_all_tokens(self) -> Iterator[Token]:
        """Iterates over all the tokens, including ones in multivariant
        blocks.
        """
        for x in self.blocks:
            match x:
                case Token():
                    yield x
                case MultiVariantBlock():
                    for option in x.options:
                        yield from option
    
    def is_timed(self) -> bool:
        """A transcription can become timed, if we fill time (in seconds)
        for all the words. This can be done with
        :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        Otherwise :code:`is_timed` is False.
        """
        return all(t.is_timed for t in self.list_all_tokens())
    
    def get_starting_part(self, time: float) -> Transcription:
        """ Cut a timed transcription up to the specified time. Is
        primarily used for streaming evaluation of partial
        transcriptions.
        
        A transcription is timed if all the tokens have their
        :attr:`~asr_eval.align.transcription.Token.start_time` and
        :attr:`~asr_eval.align.transcription.Token.end_time` filled with
        not-NaN values. The current method selects only the tokens
        up to the specified time.
        
        If :code:`time` is inside a token, converts it into a
        multivariant block with options :code:`[token]` and :code:`[]`.
        For example, let :code:`blocks = [A, B]`, token :code:`A` spans
        from 1.0 to 2.0 and :code:`B` spans from 3.0 to 4.0. Then
        :code:`get_starting_part(time=3.5)` returns
        :code:`[A, MultiVariant(X)]`, where :code:`X == [[B], []]`.
        
        If :code:`time` is inside an existing multivariant block, then
        cuts each option up to the :code:`time`, and if :code:`time` is
        inside some token in some option, add another option with this
        token excluded. For example, let
        :code:`blocks = [A, MultiVariant([[B1], [B2, B3]])]`, and
        :code:`B1` spans from 3.0 to 4.0, :code:`B2` spans from 3.0 to
        3.5, :code:`B3` spans from 3.5 to 4.0. Then
        :code:`get_starting_part(time=3.7)` returns
        :code:`[A, MultiVariant(X)]`, where
        :code:`X == [[], [B1], [B2], [B2, B3]]`. Here :code:`[]` was
        obtained from cutting option :code:`[B1]` and :code:`[B2]` was
        obtained from cutting option :code:`[B2, B3]`.
        
        Returns a copy without modifying the original object.
        """
        assert self.is_timed()
        
        blocks_partial_stem: list[Token | MultiVariantBlock] = []
        blocks_partial_tail_options: list[list[Token]] = []

        for block in self.blocks:
            if block.start_time >= time:
                break
            elif block.end_time <= time:
                blocks_partial_stem.append(block)
            else:
                # `time` is inside the block
                if isinstance(block, Token):
                    blocks_partial_tail_options.append([block])
                    blocks_partial_tail_options.append([])
                else:
                    for option in block.options:
                        option_partial = [
                            t for t in option if t.start_time < time
                        ]
                        blocks_partial_tail_options.append(option_partial)
                        if (
                            len(option_partial)
                            and option_partial[-1].end_time > time
                        ):
                            blocks_partial_tail_options.append(
                                option_partial[:-1]
                            )
                break
            
        resulting_blocks = blocks_partial_stem
    
        if len(resulting_blocks) == 0:
            end_pos = 0
        else:
            match (t := resulting_blocks[-1]):
                case Token():
                    end_pos = t.end_pos
                case MultiVariantBlock():
                    end_pos = t.end_pos
        resulting_text = self.text[:end_pos]

        if len(blocks_partial_tail_options):
            resulting_blocks.append(b := MultiVariantBlock(
                blocks_partial_tail_options,
                start_pos=0, end_pos=0,  # positions have no meaning
            ))
            resulting_text += ' ' + b.to_text()
        
        return Transcription(resulting_text, tuple(resulting_blocks))
    
    def colorize(self,  color_mode: Literal['ansi', 'html'] = 'ansi') -> str:
        """Colorizes each token in the parsed (possibly multivariant)
        string. Returns string with ANSI escape codes (rendered using
        :code:`print` in jupyter or console), or HTML color spans.
        
        See example in the :class:`~asr_eval.align.alignment.Alignment`
        docstring.
        """
        colors = [
            'on_light_yellow',
            'on_light_green',
            'on_light_cyan',
            # 'on_light_grey',
            # 'on_light_red',
            # 'on_light_blue',
            # 'on_light_magenta',
        ]
        
        token_idx = 0
        token_uid_to_color: dict[TOKEN_UID, str] = {}
        
        formatting_spans = [FormattingSpan(
            Formatting(attrs={'bold'}), 0, len(self.text)
        )]
        
        def mark_token(token: Token):
            nonlocal token_idx, token_uid_to_color
            color = colors[token_idx % len(colors)]
            formatting_spans.append(FormattingSpan(
                Formatting(on_color=color), token.start_pos, token.end_pos,
            ))
            token_uid_to_color[token.uid] = color
            token_idx += 1
            
        for block in self.blocks:
            match block:
                case Token():
                    mark_token(block)
                case MultiVariantBlock():
                    formatting_spans.append(FormattingSpan(
                        Formatting(attrs={'underline'}),
                        block.start_pos,
                        block.end_pos,
                    ))
                    for option in block.options:
                        for token in option:
                            mark_token(token)

        return apply_formatting(
            self.text, formatting_spans, color_mode=color_mode
        )
    
    def select_single_path(
        self, multivariant_choices: Sequence[int]
    ) -> TranscriptionPath:
        """Returns a transcription with the selected option in each
        multivariant block.
    
        Note:
            This is a lower-level function typically not called manually
            use :func:`~asr_eval.align.alignment.Alignment` constructor
            instead.
        
        The :code:`multivariant_choices` are usually obtained from
        :func:`~asr_eval.align.solvers.dynprog.solve_optimal_alignment`.
        The :code:`multivariant_choices` length should equal the total
        count of multivariant blocks.
        """
        return TranscriptionPath(
            text=self.text,
            blocks=self.blocks,
            multivariant_choices=tuple(multivariant_choices),
        )


@dataclass(frozen=True)
class SingleVariantTranscription(Transcription):
    """A subclass of
    :class:`~asr_eval.align.transcription.Transcription` used
    for typing clarity where we do not expect multivariance (mainly for
    predictions). Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_single_variant_transcription`.
    """
    blocks: tuple[Token, ...]


@dataclass(frozen=True)
class OuterLoc:
    """A slot that represents a specific position in the ground truth:
    before/at/after some word index.
    
    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    mod: Literal['at', 'pre']
    pos: int

@dataclass(frozen=True)
class InnerLoc(OuterLoc):
    """A slot that represents a specific inner position in the
    multivariant ground truth: before/at/after some word index
    in a multivariant option.
    
    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    inner_mod: Literal['at', 'pre']
    inner_pos: int

def _is_insertion_loc(loc: InnerLoc | OuterLoc) -> bool: # pyright: ignore[reportUnusedFunction]
    if isinstance(loc, InnerLoc):
        return loc.inner_mod == 'pre'
    else:
        return loc.mod == 'pre'

def get_outer_slots(
    blocks: Sequence[Token | MultiVariantBlock]
) -> Iterator[OuterLoc]:
    """Enumerates all the outer slots in the ground truth.
    
    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    for i in range(len(blocks)):
        yield OuterLoc('pre', i)
        yield OuterLoc('at', i)
    yield OuterLoc('pre', len(blocks))


def get_outer_slots_values(
    blocks: Sequence[Token | MultiVariantBlock]
) -> Iterator[Token | MultiVariantBlock | None]:
    """Enumerates all the outer slot values in the ground truth.
    
    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    for token in blocks:
        yield None
        yield token
    yield None


@dataclass(frozen=True)
class TranscriptionPath(Transcription):
    """ A Transcription with a selected option for each multivariant
    block.
    
    Note:
        This is a lower-level subclass that extends
        :class:`~asr_eval.align.transcription.Transcription` with a few
        indexing methods that are typically not called manually. See the
        :class:`~asr_eval.align.transcription.Transcription`
        docs for the main methods.
    
    See more details in :doc:`/guide_alignment_wer`.
    """
    
    multivariant_choices: tuple[int, ...]
    
    # internal fields that are filled in __post_init__ (do not fill manually)
    _choices_by_mvuid: dict[TOKEN_UID, int] = (
        field(default_factory=dict[TOKEN_UID, int])
    )
    _uid_to_slot_loc: dict[TOKEN_UID, OuterLoc | InnerLoc] = (
        field(default_factory=dict[TOKEN_UID, OuterLoc | InnerLoc])
    )
    _slot_locs: list[OuterLoc] = (
        field(default_factory=list[OuterLoc])
    )
    _slot_loc_to_index: dict[InnerLoc | OuterLoc, int] = (
        field(default_factory=dict[InnerLoc | OuterLoc, int])
    )
    
    def __post_init__(self):
        # checking that internal fields are not filled
        assert len(self._choices_by_mvuid) == 0
        assert len(self._uid_to_slot_loc) == 0
        assert len(self._slot_locs) == 0
        assert len(self._slot_loc_to_index) == 0
        
        # validating self.multivariant_choices
        multivariant_blocks = [
            b for b in self.blocks if isinstance(b, MultiVariantBlock)
        ]
        assert len(self.multivariant_choices) == len(multivariant_blocks), (
            f'The transcription has {len(multivariant_blocks)}'
            f' multivariant blocks'
            f', but {len(self.multivariant_choices)} choices are provided'
        )
        for block, option_idx in zip(
            multivariant_blocks, self.multivariant_choices, strict=True
        ):
            assert option_idx < len(block.options), (
                f'Option index {option_idx} is provided'
                f', but the block has only {len(block.options)} options'
            )
        
        # filling a mapping from multivariant block uid to selected option
        for block, option_idx in zip(
            multivariant_blocks, self.multivariant_choices, strict=True
        ):
            assert block.uid not in self._choices_by_mvuid, (
                'duplicate multivariant block index'
            )
            self._choices_by_mvuid[block.uid] = option_idx
        
        # filling a mapping from token uid to slot position
        for block_idx, block in enumerate(self.blocks):
            match block:
                case Token():
                    self._uid_to_slot_loc[block.uid] = (
                        OuterLoc('at', block_idx)
                    )
                case MultiVariantBlock():
                    option_idx = self._choices_by_mvuid[block.uid]
                    selected_option = block.options[option_idx]
                    for i, token in enumerate(selected_option):
                        self._uid_to_slot_loc[token.uid] = (
                            InnerLoc('at', block_idx, 'at', i)
                        )
        
        # determining slots
        self._slot_locs.append(loc := self._first_slot_loc())
        while True:
            try:
                loc = self._calc_next_slot_loc(loc)
                self._slot_locs.append(loc)
            except StopIteration:
                break
        
        # building a reverse mapping from location to index
        for i, loc in enumerate(self._slot_locs):
            self._slot_loc_to_index[loc] = i
    
    def get_prev_slot(
        self, loc: InnerLoc | OuterLoc
    ) -> InnerLoc | OuterLoc | None:
        """A step backward: from the next slot to the previous.
        
        Returns None if we reached the end.
        """
        
        idx = self.slot_loc_to_idx(loc)
        if idx == 0:
            return None
        return self._slot_locs[idx - 1]
    
    def get_next_slot(
        self, loc: InnerLoc | OuterLoc
    ) -> InnerLoc | OuterLoc | None:
        """A step forward: from the previous slot to the next.
        
        Returns None if we reached the beginning.
        """
        
        idx = self.slot_loc_to_idx(loc)
        if idx == len(self._slot_locs) - 1:
            return None
        return self._slot_locs[idx + 1]
    
    def slot_idx_to_loc(self, index: int) -> InnerLoc | OuterLoc:
        """Get a slot by slot index."""
        
        return self._slot_locs[index]
    
    def slot_loc_to_idx(self, loc: InnerLoc | OuterLoc) -> int:
        """Get a slot index for a given slot."""
        
        return self._slot_loc_to_index[loc]
    
    def token_uid_to_slot(
        self, uid: TOKEN_UID
    ) -> tuple[int, InnerLoc | OuterLoc]:
        """Get a slot and a slot index for a given Token
        :attr:`~asr_eval.align.transcription.Token.uid`.
        """
        
        try:
            loc = self._uid_to_slot_loc[uid]
            loc_idx = self._slot_loc_to_index[loc]
            return loc_idx, loc
        except KeyError:
            raise AssertionError(f'uid {uid} is not in the selected path')
    
    def slot_to_token(self, loc: InnerLoc | OuterLoc) -> Token:
        """Get a Token for a given "at" index (outer or inner)."""
        
        loc_outer = OuterLoc(loc.mod, loc.pos)
        assert loc_outer.mod == 'at', 'cannot retrieve token for "pre" loc'
        block = self.blocks[loc_outer.pos]
        match block:
            case Token():
                assert not isinstance(loc, InnerLoc), (
                    'incorrect loc: at Token, cannot have inner index'
                )
                return block
            case MultiVariantBlock():
                assert isinstance(loc, InnerLoc), (
                    'incorrect loc: at MultiVariantBlock, need inner index'
                )
                assert loc.inner_mod == 'at', 'cannot retrieve token for "pre" loc'
                option_idx = self._choices_by_mvuid[block.uid]
                selected_option = block.options[option_idx]
                assert 0 <= loc.inner_pos < len(selected_option), (
                    f'incorrect loc: bad inner index {loc.inner_pos} (at)'
                    f' for option with length {len(selected_option)}'
                )
                return selected_option[loc.inner_pos]
    
    def _first_slot_loc(self) -> OuterLoc:
        """Get the first slot."""
        
        return OuterLoc('pre', 0)
    
    def _calc_next_slot_loc(
        self, loc: OuterLoc | InnerLoc
    ) -> OuterLoc | InnerLoc:
        """Performs a transition from the previous slot to the next.
        
        Raises StopIteration if reached the end.
        """
        loc_outer = OuterLoc(loc.mod, loc.pos)
        match loc_outer.mod:
            case 'pre':
                assert not isinstance(loc, InnerLoc), (
                    'incorrect loc: outer modifier is "pre"'
                    ', cannot have inner index'
                )
                if loc_outer.pos == len(self.blocks):
                    # reached the end of the transcription
                    raise StopIteration
                next_block = self.blocks[loc_outer.pos]
                match next_block:
                    case Token():
                        # enter the token outside a multivariant block
                        return OuterLoc('at', loc_outer.pos)
                    case MultiVariantBlock():
                        option_idx = self._choices_by_mvuid[next_block.uid]
                        selected_option = next_block.options[option_idx]
                        if len(selected_option) == 0:
                            # step over the multivariant block with empty
                            # selected option
                            return OuterLoc('pre', loc_outer.pos + 1)
                        else:
                            # step into the multivariant block
                            return InnerLoc('at', loc_outer.pos, 'at', 0)
            case 'at':
                at_block = self.blocks[loc_outer.pos]
                match at_block:
                    case Token():
                        assert not isinstance(loc, InnerLoc), (
                            'incorrect loc: at Token, cannot have inner index'
                        )
                        # exit the token outside a multivariant block
                        return OuterLoc('pre', loc_outer.pos + 1)
                    case MultiVariantBlock():
                        assert isinstance(loc, InnerLoc), (
                            'incorrect loc: at MultiVariantBlock'
                            ', need inner index'
                        )
                        option_idx = self._choices_by_mvuid[at_block.uid]
                        selected_option = at_block.options[option_idx]
                        match loc.inner_mod:
                            case 'pre':
                                assert (
                                    0 < loc.inner_pos < len(selected_option)
                                ), (
                                    'incorrect loc: bad inner'
                                    f' index {loc.inner_pos} (pre)'
                                    ' for option with length'
                                    f' {len(selected_option)}'
                                )
                                # enter the next token in the multivariant
                                # block
                                return InnerLoc(
                                    'at', loc_outer.pos, 'at', loc.inner_pos
                                )
                            case 'at':
                                assert (
                                    0 <= loc.inner_pos < len(selected_option)
                                ), (
                                    'incorrect loc: bad inner index'
                                    f' {loc.inner_pos} (at)'
                                    ' for option with length'
                                    f' {len(selected_option)}'
                                )
                                if loc.inner_pos == len(selected_option) - 1:
                                    # exit the multivariant block
                                    return OuterLoc('pre', loc_outer.pos + 1)
                                else:
                                    # enter a space between tokens in the
                                    # multivariant block
                                    return InnerLoc(
                                        'at',
                                        loc_outer.pos,
                                        'pre',
                                        loc.inner_pos + 1,
                                    )