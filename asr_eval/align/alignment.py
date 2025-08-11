from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Self

from asr_eval.utils.table import Table2D

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


SLOT_VALUE = list[Correct | Replacement | Insertion | Deletion]


@dataclass
class Alignment:
    true: MultiVariantTranscriptionPath
    slots: dict[SLOT_LOC, SLOT_VALUE]
    
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
        slots: dict[SLOT_LOC, SLOT_VALUE] = defaultdict(list)
        
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
                    slot_loc = true.slot_idx_to_loc(last_true_slot_idx - 1)
                assert match.pred is not None
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
        outer_slots = get_outer_slots(self.baseline)
        outer_slot_baseline_values = get_outer_slots_values(self.baseline)
        outer_slot_to_index: dict[OUTER_LOC, int] = {loc: i for i, loc in enumerate(outer_slots)}

        outer_slot_values = Table2D[list[SLOT_VALUE]](
            rows=len(self.alignments),
            cols=len(outer_slots),
            default=list,
        )

        for row_idx, alignment in enumerate(self.alignments.values()):
            for loc, values in alignment.slots.items():
                outer_loc = loc[:2] if len(loc) == 4 else loc
                outer_slot_idx = outer_slot_to_index[outer_loc]
                outer_slot_values[row_idx, outer_slot_idx].append(values)
        
        return MultipleAlignmentView(
            baseline_words=outer_slot_baseline_values,
            names=list(self.alignments),
            values=outer_slot_values,
        )


@dataclass
class MultipleAlignmentView:
    baseline_words: list[str]
    names: list[str]
    values: Table2D[list[SLOT_VALUE]]