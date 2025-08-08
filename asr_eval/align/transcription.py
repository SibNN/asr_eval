from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
import uuid

import numpy as np

from ..utils.formatting import Formatting, FormattingSpan, apply_ansi_formatting



__all__ = [
    "Anything",
    "Token",
    "MultiVariantBlock",
]


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
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
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
    
    def __repr__(self) -> str:
        return f'MultiVariant({str(self.options)[1:-1]})'

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

    def to_text(self) -> str:
        options_str: list[str] = []
        for option in self.options:
            options_str.append(' '.join(
                ('<*>' if isinstance(t.value, Anything) else t.value)
                for t in option
            ))
        return '{' + '|'.join(options_str) + '}'
    
    
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
    
    def itertokens(self) -> Iterator[Token]:
        for x in self.tokens:
            match x:
                case Token():
                    yield x
                case MultiVariantBlock():
                    for option in x.options:
                        yield from option
    
    def is_timed(self) -> bool:
        return all(t.is_timed for t in self.itertokens())
    
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
        token_uid_to_color: dict[str, str] = {}
        
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


@dataclass
class MultiVariantTranscription(BaseTranscription[list[Token | MultiVariantBlock]]):
    pass


@dataclass
class SingleVariantTranscription(BaseTranscription[list[Token]]):
    pass