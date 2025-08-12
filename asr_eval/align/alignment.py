from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Self

from termcolor import colored

from ..utils.table import Table2D
from .matching import Match, solve_optimal_alignment
from .transcription import (
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
class Deletion:
    pass

@dataclass
class Correct:
    token: Token

@dataclass
class Replacement:
    token: Token

@dataclass
class Insertion:
    token: Token


SLOT_VALUES = list[Correct | Replacement | Insertion | Deletion]


@dataclass
class Alignment:
    '''
    TODO docstring
    '''
    true: MultiVariantTranscriptionPath
    slots: dict[SLOT_LOC, SLOT_VALUES]
    
    @classmethod
    def from_predictions(
        cls,
        true: MultiVariantTranscription | SingleVariantTranscription,
        pred: SingleVariantTranscription,
    ) -> Self:
        matches_list, multivariant_choices = solve_optimal_alignment(true.tokens, pred.tokens)
        true = true.select_single_path(multivariant_choices)
        return cls.from_matches(true, matches_list.matches)

    @classmethod
    def from_matches(
        cls,
        true: MultiVariantTranscriptionPath,
        matches: list[Match],
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
                    # deletion
                    slots[slot_loc].append(Deletion())
            else:
                # insertion
                if last_true_slot_idx is None:
                    # before the first true token
                    slot_loc = true.slot_idx_to_loc(0)
                else:
                    print(f'{last_true_slot_idx=}')
                    slot_loc = true.slot_idx_to_loc(last_true_slot_idx + 1)
                assert match.pred is not None
                print(f'Inserting {match.pred} at {slot_loc}')
                slots[slot_loc].append(Insertion(match.pred))
            
        return cls(true=true, slots=dict(slots))  # defaultdict -> dict, to be serializable


@dataclass
class MultipleAlignment:
    baseline: MultiVariantTranscription | SingleVariantTranscription
    baseline_name: str | Literal[True] = True
    alignments: dict[str, Alignment] = field(default_factory=dict)
    
    def add_alignment_from_prediction(self, name: str, pred: SingleVariantTranscription):
        self.alignments[name] = Alignment.from_predictions(self.baseline, pred)
        
    def view(self) -> MultipleAlignmentView:
        outer_slots = get_outer_slots(self.baseline.tokens)
        outer_slot_baseline_values = get_outer_slots_values(self.baseline.tokens)
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
            baseline_words=outer_slot_baseline_values,
            baseline_name=self.baseline_name,
            names=list(self.alignments),
            table=outer_slot_values,
        )


@dataclass
class MultipleAlignmentView:
    '''
    TODO docstring
    '''
    baseline_words: list[str]
    baseline_name: str | Literal[True]
    names: list[str]
    table: Table2D[SLOT_VALUES]
    
    def render_as_text(self, mode: Literal['ansi', 'html', None] = 'ansi') -> str:
        '''
        TODO docstring
        
        Example:
        from IPython.display import HTML
        HTML(multiple_alignment.view.render_as_text('html'))
        '''
        unlabeled = self.baseline_name is not True
        
        html_err_color = '#FF9C9C' if not unlabeled else '#FFDB85'
        ansi_err_on_color = 'on_yellow'  # on_red is too dark
        
        true_lengths = [len(x) for x in self.baseline_words]
        
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
            nonlocal true_lengths
            words: list[str] = []
            lengths: list[int] = []
            for x in cell:
                text = ' ' * true_lengths[col] if isinstance(x, Deletion) else x.token.to_text()
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
        
        # if unlabeled:
        #     col_has_mismatch_with_baseline = [
        #         any(
        #             any(not isinstance(value, Correct) for value in values)
        #             for values in self.table[:, col_idx]
        #         )
        #         for col_idx in range(self.table.shape[1])
        #     ]
        #     first_row = [
        #         colorize_err(word) if has_mismatch_with_baseline else (word, len(word))
        #         for word, has_mismatch_with_baseline
        #         in zip(self.baseline_words, col_has_mismatch_with_baseline)
        #     ]
        # else:
        #     first_row = [(word, len(word)) for word in self.baseline_words]
        first_row = [(word, len(word)) for word in self.baseline_words]
        table_str.prepend_row(first_row)
        
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
            lines.append(line)
            
        # use single quotes inside the string
        html_style = """white-space: pre; font-family: 'Consolas', 'Ubuntu Mono', 'Monaco', monospace"""

        match mode:
            case 'html':
                text_block = '<br/>'.join(lines)
                text_block = f'<span style="{html_style}">' + text_block + '</span>'
            case _:
                text_block = '\n'.join(lines)

        return text_block