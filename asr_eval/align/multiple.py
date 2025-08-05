import pandas as pd

from .data import MatchesList, Token
from .recursive import align


def _align_into_table(truth: list[Token], pred: MatchesList) -> list[Token | None | list[Token]]:
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
    '''
    result: list[Token | None | list[Token]] = [None] * (2 * len(truth) + 1)

    true_index = 0  # если 0, значит мы на 0м токене из truth или перед ним

    def true_index_to_pred_index(i: int) -> int:
        return 2 * i + 1

    for match in pred.matches:
        if match.true is not None:
            # соответствует какому-то токену из truth
            table_index = true_index_to_pred_index(true_index)
            result[table_index] = match.pred
            true_index += 1
        else:
            # не соответствует никакому токену из truth (т. е. вставка)
            table_index = true_index_to_pred_index(true_index) - 1
            if result[table_index] is None:
                result[table_index] = []
            result[table_index].append(match.pred) # type: ignore

    return result


def multiple_transcriptions_alignment(
    truth: list[Token],
    predictions: dict[str, MatchesList],
) -> tuple[pd.DataFrame, str]:
    rows = {}
    for title, alignment in [('truth', align(truth, truth))] + list(predictions.items()): # type: ignore
        row = _align_into_table(truth, alignment) # type: ignore
        rows[title] = row

    table = pd.DataFrame(rows).T

    def cell_to_str(cell: Token | None | list[Token]) -> str:
        if cell is None:
            return ''
        elif isinstance(cell, list):
            return ' '.join(str(t.value) for t in cell) 
        else:
            return str(cell.value)

    table = table.map(cell_to_str) # type: ignore

    table = pd.DataFrame({
        col_name: col for col_name, col in table.items() # type: ignore
        if max(len(x) for x in col.values) > 0 # type: ignore
    })
    
    return table, '\n'.join(str(table).splitlines()[1:])


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