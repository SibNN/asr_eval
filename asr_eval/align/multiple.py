import pandas as pd

from .data import MatchesList, Token
from .recursive import align


def _align_into_table(
    truth: list[Token], pred: MatchesList
) -> tuple[list[Token | None | list[Token]], list[bool]]:
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
    result: list[Token | None | list[Token]] = [None] * (2 * len(truth) + 1)
    error_positions: list[bool] = [False] * (2 * len(truth) + 1)

    true_index = 0  # if 0, we are on 0-th true token or in before it

    def true_index_to_pred_index(i: int) -> int:
        return 2 * i + 1

    for match in pred.matches:
        if match.true is not None:
            # matches some token from truth
            table_index = true_index_to_pred_index(true_index)
            result[table_index] = match.pred
            error_positions[table_index] = match.status != 'correct'
            true_index += 1
        else:
            # does not match any token in truth (insertion)
            table_index = true_index_to_pred_index(true_index) - 1
            if result[table_index] is None:
                result[table_index] = []
            result[table_index].append(match.pred) # type: ignore
            error_positions[table_index] = True

    return result, error_positions


def multiple_transcriptions_alignment(
    truth: list[Token],
    predictions: dict[str, MatchesList],
) -> tuple[pd.DataFrame, str]:
    rows: dict[str, list[Token | list[Token] | None]] = {}
    errors: dict[str, list[bool]] = {}
    for title, alignment in (
        [('truth', align(truth, truth))] # type: ignore
        + list(predictions.items())
    ):
        row, error_positions = _align_into_table(truth, alignment)
        rows[title] = row
        errors[title] = error_positions

    table = pd.DataFrame(rows).T
    error_table = pd.DataFrame(errors).T

    def cell_to_str(cell: Token | None | list[Token]) -> str:
        if cell is None:
            return ''
        elif isinstance(cell, list):
            return ' '.join(str(t.value) for t in cell) 
        else:
            return str(cell.value)

    for row_idx in range(len(table)):
        for col_idx in range(len(table.columns)):
            value = cell_to_str(table.iat[row_idx, col_idx]) # type: ignore
            if error_table.iat[row_idx, col_idx]: # type: ignore
                value = value.upper()  # we will replace this to color
            table.iat[row_idx, col_idx] = value # type: ignore

    table = pd.DataFrame({
        col_name: col for col_name, col in table.items() # type: ignore
        if max(len(x) for x in col.values) > 0 # type: ignore
    })
    
    return table, table.to_string(header=False, max_cols=9999) # type: ignore


'''
Example:

from asr_eval.align.parsing import split_text_into_tokens
from asr_eval.align.multiple import multiple_transcriptions_alignment

truth = split_text_into_tokens('раз два три четыре')
df, text = multiple_transcriptions_alignment(truth, {
    'model1': align(truth, split_text_into_tokens('раз один один два три четыре ы')), # type: ignore
    'model2': align(truth, split_text_into_tokens('а раз один два три четыре')), # type: ignore
    'model3': align(truth, split_text_into_tokens('раз два три четыре пять')), # type: ignore
})
print(text)
'''