from dataclasses import dataclass

from .matching import Match, solve_optimal_alignment
from .parsing import parse_multivariant_string, parse_single_variant_string
from .transcription import MultiVariantTranscription, SingleVariantTranscription, Token


@dataclass
class Alignment:
    '''
    TODO docstring
    '''
    truth: MultiVariantTranscription | SingleVariantTranscription
    truth_single_variant: list[Token]
    pred: SingleVariantTranscription
    multivariant_indices: list[int]
    buckets: list[list[str]]
    error_positions: list[bool]
    
    def __init__(
        self,
        truth: str | MultiVariantTranscription | SingleVariantTranscription,
        pred: str | SingleVariantTranscription,
    ):
        if isinstance(truth, str):
            truth = parse_multivariant_string(truth)
        if isinstance(pred, str):
            pred = parse_single_variant_string(pred)
        
        self.truth = truth
        self.pred = pred
        
        matches_list, self.multivariant_indices = (
            solve_optimal_alignment(truth.tokens, pred.tokens)
        )
        self.truth_single_variant = (
            truth.to_single_variant(self.multivariant_indices).tokens
        )
        self.buckets, self.error_positions = (
            _fill_attributions(self.truth_single_variant, matches_list.matches)
        )


def _fill_attributions(
    truth: list[Token], matches: list[Match]
) -> tuple[list[list[str]], list[bool]]:
    '''
    Given a truth of length N, returns a list of length N + 1
    The odd elements of the list correspond to positions in truth
    The even elements correspond to the spaces between them.

    For example, for len(truth) == 2:

    output array element 0: tokens from pred before position 1 in truth
    output array element 1: token from pred for position 1 in truth
    output array element 2: tokens from pred between position 1 and 2 in truth
    output array element 3: token from pred for position 2 in truth
    output array element 4: tokens from pred after position 2 in truth
    
    Also returns a list indicating error positions.
    '''
    n_buckets = 2 * len(truth) + 1
    result: list[list[str]] = [[] for _ in range(n_buckets)]
    error_positions: list[bool] = [False] * n_buckets

    true_index = 0  # if 0, we are on 0-th true token or in before it

    def true_index_to_pred_index(i: int) -> int:
        return 2 * i + 1

    for match in matches:
        if match.true is not None:
            # matches some token from truth
            table_index = true_index_to_pred_index(true_index)
            if match.pred is not None:
                result[table_index].append(match.pred.uid)
            error_positions[table_index] = match.status != 'correct'
            true_index += 1
        else:
            # does not match any token in truth (insertion)
            table_index = true_index_to_pred_index(true_index) - 1
            result[table_index].append(match.pred.uid) # type: ignore
            error_positions[table_index] = True

    return result, error_positions