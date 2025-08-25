from __future__ import annotations

from collections import defaultdict
from collections.abc import Container
from dataclasses import dataclass, field, replace
from typing import Literal, Self, cast

import numpy as np
import pandas as pd
from termcolor import colored

from .metrics import Metrics
from ..utils.table import Table2D
from .matching import Match, char_edit_distance, select_shortest_multi_variants, solve_optimal_alignment
from .transcription import (
    MultiVariantBlock,
    MultiVariantTranscription,
    MultiVariantTranscriptionPath,
    SingleVariantTranscription,
    Token,
    SLOT_LOC,
    OUTER_LOC,
    get_outer_slots,
    get_outer_slots_values,
)


@dataclass
class TokenContainerMixin:
    token: Token
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.token!r})'


class Deletion:
    def __repr__(self) -> str:
        return 'Deletion()'


class Correct(TokenContainerMixin):
    pass


class Replacement(TokenContainerMixin):
    pass


class Insertion(TokenContainerMixin):
    pass


SLOT_VALUES = list[Correct | Replacement | Insertion | Deletion]


@dataclass
class ErrorListingElement:
    '''
    An outer slot when a model made a mistake.
    '''
    outer_loc: OUTER_LOC
    true: Token | MultiVariantBlock | None
    true_text: str | None
    pred: SLOT_VALUES
    n_replacements: int
    n_insertions: int
    n_deletions: int
    sample_idx: int | None = None
    
    @property
    def n_errors(self) -> int:
        return self.n_replacements + self.n_insertions + self.n_deletions
    
    @property
    def pred_text(self) -> str:
        return ' '.join([('' if isinstance(x, Deletion) else x.token.to_text()) for x in self.pred])


@dataclass
class Alignment:
    '''
    TODO docstring
    '''
    true: MultiVariantTranscriptionPath
    pred: SingleVariantTranscription
    slots: dict[SLOT_LOC, SLOT_VALUES]
    
    def error_listing(
        self,
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
    ) -> tuple[list[ErrorListingElement], Metrics]:
        err_positions_dict: dict[OUTER_LOC, ErrorListingElement] = {}
        for loc, slot_values in self.slots.items():
            assert len(slot_values)
            
            outer_mod, outer_idx = outer_loc = loc[:2]
            if not outer_loc in err_positions_dict:
                true = self.true.tokens[outer_idx] if outer_mod == 'at' else None
                err_positions_dict[outer_loc] = ErrorListingElement(
                    outer_loc=outer_loc,
                    true=true,
                    true_text=true.to_text() if true is not None else None,
                    pred=[],
                    n_replacements=0,
                    n_insertions=0,
                    n_deletions=0,
                )
            
            err_pos = err_positions_dict[outer_loc]
            err_pos.pred += slot_values
            
            # count <Correct | Replacement | Insertion | Deletion> per cell
            slot_replacements = sum(isinstance(x, Replacement) for x in slot_values)
            slot_insertions = sum(isinstance(x, Insertion) for x in slot_values)
            slot_deletions = sum(isinstance(x, Deletion) for x in slot_values)
            slot_correct = sum(isinstance(x, Correct) for x in slot_values)
            
            # determine cell type
            has_replacements = slot_replacements > 0
            has_insertions = slot_insertions > 0
            has_deletions = slot_deletions > 0
            has_corrects = slot_correct > 0
            
            # validate cell type
            if has_replacements:
                # slot of "replacement" type, may only contain absorbed insertions
                assert not has_deletions and not has_corrects
            else:
                # slot of "deletion", "insertion", or "correct" type
                assert sum((has_insertions, has_deletions, has_corrects)) == 1
        
            # count replacements, deletions
            err_pos.n_replacements += slot_replacements
            err_pos.n_deletions += slot_deletions
            
            # count insertions
            if has_replacements:
                # for "replacement" cells, we may count or not count absorbed insertions
                if count_absorbed_insertions:
                    err_pos.n_insertions += slot_insertions
            elif max_consecutive_insertions is not None:
                # for "insertion" cell, set an upper bound for slot insertions count
                err_pos.n_insertions += min(slot_insertions, max_consecutive_insertions)
            else:
                err_pos.n_insertions += slot_insertions
        
        err_positions = [pos for pos in err_positions_dict.values() if pos.n_errors > 0]
        
        metrics = Metrics(
            true_len=self.get_true_len(),
            n_replacements=sum([pos.n_replacements for pos in err_positions]),
            n_insertions=sum([pos.n_insertions for pos in err_positions]),
            n_deletions=sum([pos.n_deletions for pos in err_positions]),
        )
        return err_positions, metrics
          
    def get_true_len(self) -> int:
        return len(select_shortest_multi_variants(self.true.tokens))
    
    # def to_outer_slots(self) -> dict[OUTER_LOC, SLOT_VALUES]:
    #     result: dict[OUTER_LOC, SLOT_VALUES] = defaultdict(list)
    #     for loc, value in self.slots.items():
    #         result[loc[:2]] += value
    #     return dict(result)
    
    @classmethod
    def from_predictions(
        cls,
        true: MultiVariantTranscription | SingleVariantTranscription,
        pred: SingleVariantTranscription,
        absorb_insertions: bool = True,
    ) -> Self:
        matches_list, multivariant_choices = solve_optimal_alignment(true.tokens, pred.tokens)
        true = true.select_single_path(multivariant_choices)
        return cls.from_matches(true, pred, matches_list.matches, absorb_insertions=absorb_insertions)

    @classmethod
    def from_matches(
        cls,
        true: MultiVariantTranscriptionPath,
        pred: SingleVariantTranscription,
        matches: list[Match],
        absorb_insertions: bool = True,
    ) -> Self:
        slots: dict[SLOT_LOC, SLOT_VALUES] = defaultdict(list)
        
        last_true_slot_idx: int | None = None
        for match in matches:
            if match.true is not None:
                slot_idx, slot_loc = true.token_uid_to_slot(match.true.uid)
                last_true_slot_idx = slot_idx
                if match.pred is not None:
                    if match.status == 'correct':
                        # correct
                        slots[slot_loc].append(Correct(match.pred))
                    else:
                        # replacement
                        assert match.status == 'replacement'
                        slots[slot_loc].append(Replacement(match.pred))
                else:
                    if match.status != 'correct':  # may be correct for unmatched Anything()
                        # deletion
                        slots[slot_loc].append(Deletion())
            else:
                # insertion
                if last_true_slot_idx is None:
                    # before the first true token
                    slot_loc = true.slot_idx_to_loc(0)
                else:
                    slot_loc = true.slot_idx_to_loc(last_true_slot_idx + 1)
                assert match.pred is not None
                slots[slot_loc].append(Insertion(match.pred))
        
        if absorb_insertions:
            absorb_insertions_into_replacements_inplace(true, slots)
            
        return cls(true=true, pred=pred, slots=dict(slots))  # defaultdict -> dict, to be serializable


def absorb_insertions_into_replacements_inplace(
    true: MultiVariantTranscriptionPath,
    slots: dict[SLOT_LOC, SLOT_VALUES]
):
    '''
    TODO docstring
    '''
    slots_becoming_empty: set[SLOT_LOC] = set()
    for slot_loc, slot_values in slots.items():
        if len(slot_values) == 1 and isinstance(slot_values[0], Replacement):
            true_text = cast(str, true.slot_to_token(slot_loc).value)
            text = cast(str, slot_values[0].token.value)
            
            # try to absorb insertions from the left side
            prev_slot = true.get_prev_slot(slot_loc)
            if prev_slot is not None and prev_slot in slots:
                prev_values = slots[prev_slot]
                for i in np.arange(len(prev_values))[::-1]:
                    if not isinstance(prev_values[i], Insertion):
                        break
                    prev_text = cast(str, prev_values[i].token.value) # type: ignore
                    if (
                        char_edit_distance(true_text, prev_text + ' ' + text)
                        < char_edit_distance(true_text, text)
                    ):
                        slot_values.insert(0, prev_values.pop())
                        text = prev_text + ' ' + text
                if len(prev_values) == 0:
                    slots_becoming_empty.add(prev_slot)
            
            # try to absorb insertions from the right side
            next_slot = true.get_next_slot(slot_loc)
            if next_slot is not None and next_slot in slots:
                next_values = slots[next_slot]
                for i in np.arange(len(next_values)):
                    if not isinstance(next_values[0], Insertion):
                        break
                    next_text = cast(str, next_values[0].token.value) # type: ignore
                    if (
                        char_edit_distance(true_text,  text + ' ' + next_text)
                        < char_edit_distance(true_text, text)
                    ):
                        slot_values.append(next_values.pop(0))
                        text = text + ' ' + next_text
                if len(next_values) == 0:
                    slots_becoming_empty.add(next_slot)
    
    for slot_loc in slots_becoming_empty:
        del slots[slot_loc]
    


@dataclass
class MultipleAlignment:
    '''
    TODO docstring
    '''
    baseline: MultiVariantTranscription | SingleVariantTranscription
    baseline_name: str | Literal[True] = True
    alignments: dict[str, Alignment] = field(default_factory=dict)
    
    def add_alignment_from_prediction(self, name: str, pred: SingleVariantTranscription):
        self.alignments[name] = Alignment.from_predictions(self.baseline, pred)
    
    def get_names(self, names: Container[str]) -> Self:
        return replace(self, alignments={
            name: alignment for name, alignment in self.alignments.items()
            if name in names
        })
        
    def view(self) -> MultipleAlignmentView:
        outer_slots = get_outer_slots(self.baseline.tokens)
        outer_slot_to_index: dict[OUTER_LOC, int] = {loc: i for i, loc in enumerate(outer_slots)}

        outer_slot_values = Table2D[SLOT_VALUES].construct(
            rows=len(self.alignments),
            cols=len(outer_slots),
            default=list,
        )

        for row_idx, alignment in enumerate(self.alignments.values()):
            for loc, values in alignment.slots.items():
                outer_loc = loc[:2] if len(loc) == 4 else loc
                outer_slot_idx = outer_slot_to_index[outer_loc]
                outer_slot_values[row_idx, outer_slot_idx] += values
        
        return MultipleAlignmentView(
            baseline=self.baseline,
            baseline_blocks=get_outer_slots_values(self.baseline.tokens),
            baseline_name=self.baseline_name,
            names=list(self.alignments),
            texts=[al.pred for al in self.alignments.values()],
            table=outer_slot_values,
        )


@dataclass
class MultipleAlignmentView:
    '''
    TODO docstring
    '''
    baseline: MultiVariantTranscription | SingleVariantTranscription
    baseline_blocks: list[Token | MultiVariantBlock | None]
    baseline_name: str | Literal[True]
    names: list[str]
    texts: list[SingleVariantTranscription]
    table: Table2D[SLOT_VALUES]
    
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=[x.to_text() if x is not None else '<gap>' for x in self.baseline_blocks],
            data=self.table.to_numpy(),
        )
        df['name'] = self.names
        df = df.set_index('name') # type: ignore
        return df
    
    def render_as_text(
        self,
        mode: Literal['ansi', 'html', None] = 'ansi',
        prefixes: list[str] | None = None,
        html_add_style: bool = True,
        add_pipeline_names: bool = True,
    ) -> str:
        '''
        TODO docstring
        
        Example:
        from IPython.display import HTML
        HTML(multiple_alignment.view.render_as_text('html'))
        '''
        unlabeled = self.baseline_name is not True
        
        html_err_color = '#FF9C9C' if not unlabeled else '#FFDB85'
        ansi_err_on_color = 'on_yellow'  # on_red is too dark
        
        # length of colored spans for Deletion()
        baseline_words = [
            self.baseline.text[block.start_pos:block.end_pos] if block is not None else ''
            for block in self.baseline_blocks
        ]
        baseline_word_lengths = [len(w) for w in baseline_words]
        
        def colorize_err(text: str) -> tuple[str, int]:
            nonlocal mode, html_err_color, ansi_err_on_color
            match mode:
                case None:
                    return text, len(text)
                case 'html':
                    return (
                        f'<span style="background-color: {html_err_color};">'
                        + text
                        + '</span>'
                    ), len(text)
                case 'ansi':
                    return colored(text, on_color=ansi_err_on_color), len(text)

        def render_cell(
            row: int,
            col: int,
            cell: SLOT_VALUES,
        ) -> tuple[str, int]:
            # returns (text, text_len) tuple for a cell
            # text_len is a count of printable characters in the text,
            # excluding ANSI color codes and HTML tags
            nonlocal baseline_word_lengths, self
            words: list[str] = []
            lengths: list[int] = []
            for x in cell:
                text = (
                    ' ' * baseline_word_lengths[col]
                    if isinstance(x, Deletion)
                    else self.texts[row].text[x.token.start_pos:x.token.end_pos]
                )
                text_len = len(text)
                if not isinstance(x, Correct):
                    text, text_len = colorize_err(text)
                words.append(text)
                lengths.append(text_len)
            return (
                ' '.join(words),
                sum(lengths) + max(0, len(lengths) - 1)
            )

        # table_str keeps (text, text_len) tuple in each cell
        table_str: Table2D[tuple[str, int]] = self.table.map_with_indices(render_cell)
        
        first_row = list(zip(baseline_words, baseline_word_lengths))
        table_str.prepend_row(first_row)
        
        if add_pipeline_names:
            table_str.prepend_col([('|', 1) for _ in range(table_str.shape[0])])
            table_str.prepend_col([(x, len(x)) for x in [str(self.baseline_name)] + self.names])

        col_lengths = [
            max([l for _, l in table_str[:, col_idx]])
            for col_idx in range(table_str.shape[1])
        ]

        lines: list[str] = []
        for row_idx in range(table_str.shape[0]):
            line = ' '.join([
                text + ' ' * (length - text_len)
                for (text, text_len), length in zip(table_str[row_idx, :], col_lengths)
            ])
            if prefixes is not None:
                line = prefixes[row_idx] + line
            lines.append(line)
            
        # use single quotes inside the string
        html_style = """white-space: pre; font-family: 'Consolas', 'Ubuntu Mono', 'Monaco', monospace"""

        match mode:
            case 'html':
                text_block = '<br/>'.join(lines)
                if html_add_style:
                    text_block = f'<span style="{html_style}">' + text_block + '</span>'
            case _:
                text_block = '\n'.join(lines)

        return text_block