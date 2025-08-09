from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast
import uuid

import numpy as np

from ..utils.formatting import Formatting, FormattingSpan, apply_ansi_formatting



__all__ = [
    "Anything",
    "Token",
    "MultiVariantBlock",
]


TOKEN_UID = str


@dataclass(slots=True)
class Anything:
    """
    Represents <*> in a multivariant transcription: a symbol that matches every word sequence or nothing.
    """
    def __eq__(self, other: Any) -> bool:
        return True


@dataclass(slots=True)
class Token:
    """
    Either a word, or `Anything` in a transcription. Additional fields:
    
    - `uid`: a unique id for the current token (is filled automatically if not specified). Useful in
    string alignments, because there can be multiple equal words in both texts, and without unique IDs
    the alignment will be ambiguous. We could use positions instead of unqiue IDs, but positions are
    also ambiguous in a multivariant transcription.
    - `start_pos` and `end_pos`: position in the original text: start (inclusive) and end (exclusive)
    characters. May be useful for displaying an alignment.
    - `start_time` and `end_time`: start and end time in seconds, if known.
    - `type` is either "word", or "punct" (see `split_text_into_tokens`), or any user-defined types
    """
    value: str | Anything
    uid: TOKEN_UID = field(default_factory=lambda: str(uuid.uuid4()))
    start_pos: int = 0
    end_pos: int = 0
    start_time: float = np.nan
    end_time: float = np.nan
    type: str = 'word'
    
    def __repr__(self) -> str:
        strings = [
            str(self.value),
            # f'pos=({self.start_pos}, {self.end_pos})'
        ]
        if self.is_timed:
            strings.append(f't=({self.start_time:.1f}, {self.end_time:.1f})')
        if self.type != 'word':
            strings.append(self.type)
        return f'Token(' + ', '.join(strings) + ')'

    @property
    def is_timed(self) -> bool:
        return not np.isnan(self.start_time) and not np.isnan(self.end_time)


@dataclass(slots=True)
class MultiVariantBlock:
    """
    A multivariant block in a transcription. Keeps a list of variants, each variant is a list of Token.
    
    `pos` represents a position in the original text, including braces {}.
    """
    options: list[list[Token]]
    pos: tuple[int, int] = (0, 0)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __repr__(self) -> str:
        return f'MultiVariantBlock({str(self.options)[1:-1]})'

    @property
    def is_timed(self) -> bool:
        return all(all(t.is_timed for t in option) for option in self.options)
    
    @property
    def start_time(self) -> float:
        """
        Will return the earliest .start_time across all options, or NaN if tokens are not timed
        """
        start_times = [option[0].start_time for option in self.options if len(option)]
        assert len(start_times), 'All options are empty in a MultiVariant block, should not happen'
        return np.min(start_times)  # `min` builtin works incorrectly: min(-1.0, np.nan) --> -1.0
    
    @property
    def end_time(self) -> float:
        """
        Will return the latest .end_time across all options, or NaN if tokens are not timed
        """
        end_times = [option[-1].end_time for option in self.options if len(option)]
        assert len(end_times), 'All options are empty in a MultiVariant block, should not happen'
        return np.max(end_times)
    
    def get_option_text(self, option_index: int) -> str:
        return ' '.join(
            ('<*>' if isinstance(t.value, Anything) else t.value)
            for t in self.options[option_index]
        )

    def to_text(self) -> str:
        return '{' + '|'.join([
            self.get_option_text(i) for i in range(len(self.options))
        ]) + '}'
    
    
T = TypeVar('T', list[Token], list[Token | MultiVariantBlock])
    

@dataclass
class BaseTranscription(Generic[T]):
    '''
    A generic for two subclasses: MultiVariantTranscription and SingleVariantTranscription.
    
    Details: Multivariant and single-variant transcription should share the same methods, but should not be
    subclasses of each other, for clarity. We also cannot define `tokens: list[Token | MultiVariantBlock]`
    in the base class, because we cannot override it with `list[Token]` in a subclass due to `list` being
    invariant. So, we use pattern with generics.
    '''
    text: str
    tokens: T
    
    def list_all_tokens(self) -> Iterator[Token]:
        for x in self.tokens:
            match x:
                case Token():
                    yield x
                case MultiVariantBlock():
                    for option in x.options:
                        yield from option
    
    def is_timed(self) -> bool:
        return all(t.is_timed for t in self.list_all_tokens())
    
    def get_starting_part(self, time: float) -> MultiVariantTranscription:
        '''
        Cut a timed multivariant transcription up to the specified time.
        Returns a copy without modifying the original object.
        
        If `time` is inside some token, then converts it into a multivariant
        block with options [token] and [].
        
        For example, let tokens = [A, B], token A spans from 1.0 to 2.0 and B
        spans from 3.0 to 4.0. Then `get_starting_part(tokens, time=3.5)` returns
        [A, MultiVariant(X)], where X = [[B], []].
        
        If `time` is inside an existing multivariant block, then cut each option
        up to the `time`, and if `time` is inside some token in some option, add
        another option with this token excluded.
        
        For example, let tokens = [A, MultiVariant([[B1], [B2, B3]])], and B1 spans
        from 3.0 to 4.0, B2 spans from 3.0 to 3.5, B3 spans from 3.5 to 4.0. Then
        `get_starting_part(time=3.7)` returns [A, MultiVariant(X)], where
        X = [[], [B1], [B2], [B2, B3]]. Here [] was obtained from cutting option
        [B1] and [B2] was obtained from cutting option [B2, B3].
        '''
        assert self.is_timed()
        
        tokens_partial_stem: list[Token | MultiVariantBlock] = []
        tokens_partial_tail_options: list[list[Token]] = []

        for block in self.tokens:
            if block.start_time >= time:
                break
            elif block.end_time <= time:
                tokens_partial_stem.append(block)
            else:
                # `time` is inside the block
                if isinstance(block, Token):
                    tokens_partial_tail_options.append([block])
                    tokens_partial_tail_options.append([])
                else:
                    for option in block.options:
                        option_partial = [t for t in option if t.start_time < time]
                        tokens_partial_tail_options.append(option_partial)
                        if len(option_partial) and option_partial[-1].end_time > time:
                            tokens_partial_tail_options.append(option_partial[:-1])
                break
            
        resulting_tokens = tokens_partial_stem
    
        if len(resulting_tokens) == 0:
            end_pos = 0
        else:
            match (t := resulting_tokens[-1]):
                case Token():
                    end_pos = t.end_pos
                case MultiVariantBlock():
                    end_pos = t.pos[-1]
        resulting_text = self.text[:end_pos]

        if len(tokens_partial_tail_options):
            resulting_tokens.append(b := MultiVariantBlock(tokens_partial_tail_options))
            resulting_text += ' ' + b.to_text()
        
        return MultiVariantTranscription(resulting_text, resulting_tokens)
    
    def colorize(self) -> str:
        '''
        Colorizes each token in the parsed (possibly multivariant) string. Returns
        string with ANSI escape codes (can be rendered using print() in jupyter or console).
        
        Example:
        
        ```
        orig_text = '{emm} at {fourty nine|49} <*> year'
        tokens = parse_multivariant_string(orig_text)
        colored_str = colorize_parsed_string(orig_text, tokens)
        print(colored_str)
        ```
        '''
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
            
        for block in self.tokens:
            match block:
                case Token():
                    mark_token(block)
                case MultiVariantBlock():
                    formatting_spans.append(FormattingSpan(
                        Formatting(attrs={'underline'}), block.pos[0], block.pos[1]
                    ))
                    for option in block.options:
                        for token in option:
                            mark_token(token)

        return apply_ansi_formatting(self.text, formatting_spans)
    
    def select_single_path(self, multivariant_choices: list[int]) -> MultiVariantTranscriptionWithPath:
        return MultiVariantTranscriptionWithPath(
            text=self.text,
            tokens=cast(list[Token | MultiVariantBlock], self.tokens),
            multivariant_choices=multivariant_choices,
        )
    
    @dataclass
    class FlatView:
        positions: list[str]
        transitions: list[list[int]]
        resolved_multivariant_blocks: dict[tuple[int, int], list[tuple[int, int]]]
    
    def flat_view(self) -> FlatView:
        '''
        TODO support a flat view and use it in the `solve_optimal_alignment`. A flat view is
        - list of token uids
        - for each flat position, list of allowed transitions
        - dict from transition (idx1, idx2) to a list of resolved multivariant blocks and options
        '''
        raise NotImplementedError


@dataclass
class MultiVariantTranscription(BaseTranscription[list[Token | MultiVariantBlock]]):
    '''
    A transcription with zero or more multivariant blocks.
    '''
    pass
                    

@dataclass
class SingleVariantTranscription(BaseTranscription[list[Token]]):
    '''
    A transcription without multivariant blocks.
    '''
    pass


MOD = Literal['at', 'pre']
OUTER_LOC = tuple[MOD, int]
INNER_LOC = tuple[MOD, int, MOD, int]
SLOT_LOC = INNER_LOC | OUTER_LOC


class MultiVariantTranscriptionWithPath(MultiVariantTranscription):
    '''
    A MultiVariantTranscription with a selected path, i. e. a selected option for each
    multivariant block.
    '''
    
    def __init__(
        self,
        text: str,
        tokens: list[Token | MultiVariantBlock],
        multivariant_choices: list[int],
    ):
        self.text = text
        self.tokens = tokens
        self._multivariant_choices = multivariant_choices
        
        # validating
        multivariant_blocks = [b for b in self.tokens if isinstance(b, MultiVariantBlock)]
        assert len(self._multivariant_choices) == len(multivariant_blocks), (
            f'The transcription has {len(multivariant_blocks)} multivariant blocks'
            f', but {len(self._multivariant_choices)} choices are provided'
        )
        for block, option_idx in zip(multivariant_blocks, self._multivariant_choices, strict=True):
            assert option_idx < len(block.options), (
                f'Option index {option_idx} is provided'
                f', but the block has only {len(block.options)} options'
            )
        
        # filling a mapping from multivariant block uid to selected option
        self._choices_by_mvuid: dict[str, int] = {}
        for block, option_idx in zip(multivariant_blocks, self._multivariant_choices, strict=True):
            assert block.uid not in self._choices_by_mvuid, 'duplicate multivariant block index'
            self._choices_by_mvuid[block.uid] = option_idx
        
        # filling a mapping from token uid to slot position
        self._uid_to_slot_loc: dict[TOKEN_UID, SLOT_LOC] = {}
        for block_idx, block in enumerate(self.tokens):
            match block:
                case Token():
                    self._uid_to_slot_loc[block.uid] = ('at', block_idx)
                case MultiVariantBlock():
                    option_idx = self._choices_by_mvuid[block.uid]
                    selected_option = block.options[option_idx]
                    for i, token in enumerate(selected_option):
                        self._uid_to_slot_loc[token.uid] = ('at', block_idx, 'at', i)
        
        # determining slots
        self._slot_locs = [loc := self._first_slot_loc()]
        while True:
            try:
                loc = self._calc_next_slot_loc(loc)
                self._slot_locs.append(loc)
            except StopIteration:
                break
        
        # building a reverse mapping from location to index
        self._slot_loc_to_index: dict[SLOT_LOC, int] = {
            loc: i for i, loc in enumerate(self._slot_locs)
        }
    
    def path(self) -> Iterator[Token]:
        multivariant_choices = self._multivariant_choices.copy()
        for block in self.tokens:
            match block:
                case Token():
                    yield block
                case MultiVariantBlock():
                    option_index = multivariant_choices.pop(0)
                    yield from block.options[option_index]
    
    def get_block(self, index: int) -> Token | list[Token]:
        block = self.tokens[index]
        match block:
            case Token():
                return block
            case MultiVariantBlock():
                option_idx = self._choices_by_mvuid[block.uid]
                return block.options[option_idx]
    
    def get_n_slots(self) -> int:
        return len(self._slot_locs)
    
    def slot_idx_to_loc(self, index: int) -> SLOT_LOC:
        return self._slot_locs[index]
    
    def slot_loc_to_idx(self, loc: SLOT_LOC) -> int:
        return self._slot_loc_to_index[loc]
    
    def token_uid_to_slot(self, uid: TOKEN_UID) -> tuple[int, SLOT_LOC]:
        try:
            loc = self._uid_to_slot_loc[uid]
            loc_idx = self._slot_loc_to_index[loc]
            return loc_idx, loc
        except KeyError:
            raise AssertionError(f'uid {uid} is not in the selected path')
    
    def slot_to_token(self, loc: SLOT_LOC) -> Token:
        mod_outer, idx_outer = loc[:2]
        assert mod_outer == 'at', 'cannot retrieve token for "pre" loc'
        block = self.tokens[idx_outer]
        match block:
            case Token():
                assert len(loc) == 2, (
                    'incorrect loc: at Token, cannot have inner index'
                )
                return block
            case MultiVariantBlock():
                assert len(loc) == 4, (
                    'incorrect loc: at MultiVariantBlock, need inner index'
                )
                mod_inner, idx_inner = loc[2:]
                assert mod_inner == 'at', 'cannot retrieve token for "pre" loc'
                option_idx = self._choices_by_mvuid[block.uid]
                selected_option = block.options[option_idx]
                assert 0 <= idx_inner < len(selected_option), (
                    f'incorrect loc: bad inner index {idx_inner} (at)'
                    f' for option with length {len(selected_option)}'
                )
                return selected_option[idx_inner]
    
    def _first_slot_loc(self) -> SLOT_LOC:
        return 'pre', 0
    
    def _calc_next_slot_loc(self, loc: SLOT_LOC) -> SLOT_LOC:
        mod_outer, idx_outer = loc[:2]
        match mod_outer:
            case 'pre':
                assert len(loc) == 2, (
                    'incorrect loc: outer modifier is "pre", cannot have inner index'
                )
                if idx_outer == len(self.tokens):
                    # reached the end of the transcription
                    raise StopIteration
                next_block = self.tokens[idx_outer]
                match next_block:
                    case Token():
                        # enter the token outside a multivariant block
                        return 'at', idx_outer
                    case MultiVariantBlock():
                        option_idx = self._choices_by_mvuid[next_block.uid]
                        selected_option = next_block.options[option_idx]
                        if len(selected_option) == 0:
                            # step over the multivariant block with empty selected option
                            return 'pre', idx_outer + 1
                        else:
                            # step into the multivariant block
                            return 'at', idx_outer, 'at', 0
            case 'at':
                at_block = self.tokens[idx_outer]
                match at_block:
                    case Token():
                        assert len(loc) == 2, 'incorrect loc: at Token, cannot have inner index'
                        # exit the token outside a multivariant block
                        return 'pre', idx_outer + 1
                    case MultiVariantBlock():
                        assert len(loc) == 4, 'incorrect loc: at MultiVariantBlock, need inner index'
                        mod_inner, idx_inner = loc[2:]
                        option_idx = self._choices_by_mvuid[at_block.uid]
                        selected_option = at_block.options[option_idx]
                        match mod_inner:
                            case 'pre':
                                assert 0 < idx_inner < len(selected_option), (
                                    f'incorrect loc: bad inner index {idx_inner} (pre)'
                                    f' for option with length {len(selected_option)}'
                                )
                                # enter the next token in the multivariant block
                                return 'at', idx_outer, 'at', idx_inner
                            case 'at':
                                assert 0 <= idx_inner < len(selected_option), (
                                    f'incorrect loc: bad inner index {idx_inner} (at)'
                                    f' for option with length {len(selected_option)}'
                                )
                                if idx_inner == len(selected_option) - 1:
                                    # exit the multivariant block
                                    return 'pre', idx_outer + 1
                                else:
                                    # enter a space between tokens in the multivariant block
                                    return 'at', idx_outer, 'pre', idx_inner + 1