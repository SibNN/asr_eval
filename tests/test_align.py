from typing import Callable
import nltk
import numpy as np
import pytest

from asr_eval.align.matching import MatchesList
from asr_eval.align.transcription import MultiVariantBlock, Transcription, SingleVariantTranscription, Token
from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.solvers import recursive, dynprog


SOLVER_TYPE = Callable[
    [Transcription, SingleVariantTranscription],
    tuple[MatchesList, list[int]]
]

@pytest.mark.parametrize('solve_optimal_alignment', [
    recursive.solve_optimal_alignment, dynprog.solve_optimal_alignment,
])
def test_edit_distance(solve_optimal_alignment: SOLVER_TYPE):
    rng = np.random.default_rng(0)
    
    for _ in range(100):
        true = [str(x) for x in rng.integers(low=0, high=3, size=rng.integers(low=0, high=15))] # type: ignore
        pred = [str(x) for x in rng.integers(low=0, high=3, size=rng.integers(low=0, high=15))] # type: ignore
        assert (
            solve_optimal_alignment(
                SingleVariantTranscription('', tuple(Token(x) for x in true)),
                SingleVariantTranscription('', tuple(Token(x) for x in pred))
            )[0].score.n_word_errors
            == nltk.edit_distance(true, pred) # pyright: ignore[reportUnknownMemberType]
        )


@pytest.mark.parametrize('solve_optimal_alignment', [
    recursive.solve_optimal_alignment, dynprog.solve_optimal_alignment,
])
def test_align_recursive(solve_optimal_alignment: SOLVER_TYPE):
    true_text = 'a <*> b c {x|y|d} {qaz|} {a|b}'
    pred_text = 'a b x y a a'
    true = DEFAULT_PARSER.parse_transcription(true_text)
    pred = DEFAULT_PARSER.parse_single_variant_transcription(pred_text)
    matches_list, _selected_multivariant_blocks = solve_optimal_alignment(true, pred)

    assert [
        (
            match.true.value if match.true is not None else None,
            match.pred.value if match.pred is not None else None,
        )
        for match in matches_list.matches
    ] == [
        ('a', 'a'),
        ('<*>', None),
        ('b', 'b'),
        ('c', 'x'),
        ('y', 'y'),
        ('a', 'a'),
        (None, 'a')
    ]

    for x in true.list_all_tokens():
        assert true_text[x.start_pos:x.end_pos] == x.value

    for x in pred.blocks:
        assert isinstance(x, Token)
        assert pred_text[x.start_pos:x.end_pos] == x.value


@pytest.mark.parametrize('solve_optimal_alignment', [
    recursive.solve_optimal_alignment, dynprog.solve_optimal_alignment,
])
def test_multivarint_block_resolution(solve_optimal_alignment: SOLVER_TYPE):
    true = DEFAULT_PARSER.parse_transcription(
        '{седьмого|7} - {восьмого|8} мая {|в} {Пуэрто-Рико} прошёл {шестнадцатый|16-й|16й|16} этап'
    )
    pred = DEFAULT_PARSER.parse_single_variant_transcription(
        'седьмого 8 мая прошел 16й этап'
    )

    _matches, selected_multivariant_blocks = solve_optimal_alignment(true, pred)
    assert selected_multivariant_blocks == [0, 1, 0, 1, 2]


@pytest.mark.parametrize('solve_optimal_alignment', [
    recursive.solve_optimal_alignment, dynprog.solve_optimal_alignment,
])
def test_align_recursive_manual_construction(solve_optimal_alignment: SOLVER_TYPE):
    true = [
        MultiVariantBlock([
            [Token('седьмого', uid='0')],
            [Token('7', uid='1')],
        ]),
        MultiVariantBlock([
            [Token('восьмого', uid='2')],
            [Token('8', uid='3')],
        ]),
        Token('мая', uid='4'),
        Token('в', uid='5'),
        Token('пуэрто', uid='6'),
        Token('рико', uid='7'),
        Token('прошел', uid='8'),
        MultiVariantBlock([
            [Token('шестнадцатый', uid='9')],
            [Token('16', uid='10'), Token('й', uid='11')],
            [Token('16й', uid='12')],
            [Token('16', uid='13')],
        ]),
        Token('этап', uid='14'),
        MultiVariantBlock([
            [Token('формулы', uid='15'), Token('1', uid='16')],
            [Token('формулы', uid='17')],
            [Token('формулы', uid='18'), Token('один', uid='19')],
            [Token('формулы', uid='20')],
        ])
    ]
    pred = [
        Token('седьмого', uid='p0'),
        Token('восьмого', uid='p1'),
        Token('мая', uid='p2'),
        Token('в', uid='p3'),
        Token('пуэрто', uid='p4'),
        Token('рико', uid='p5'),
        Token('прошел', uid='p6'),
        Token('шестнадцатый', uid='p7'),
        Token('этап', uid='p8'),
        Token('формулы', uid='p9'),
    ]

    matches, blocks = solve_optimal_alignment(
        Transcription('', tuple(true)),
        SingleVariantTranscription('', tuple(pred)),
    )

    assert matches.score.n_correct == 10
    b1, b2, b3, b4 = blocks
    assert [b1, b2, b3] == [0, 0, 0]
    assert b4 in (1, 3)