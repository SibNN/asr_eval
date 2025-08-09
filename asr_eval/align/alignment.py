from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from .matching import Match, solve_optimal_alignment
from .transcription import (
    MultiVariantTranscription,
    MultiVariantTranscriptionWithPath,
    SingleVariantTranscription,
    Token,
    SLOT_LOC,
    OUTER_LOC,
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
    true: MultiVariantTranscriptionWithPath
    slots: dict[SLOT_LOC, SLOT_VALUE]
    
    def groupby_outer(self) -> dict[OUTER_LOC, SLOT_VALUE]:
        result: dict[OUTER_LOC, SLOT_VALUE] = defaultdict(list)
        for loc, values in self.slots.items():
            result[loc[:2]] += values
        return result
    
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
        true: MultiVariantTranscriptionWithPath,
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