import nltk # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from asr_eval.align.data import Token
from asr_eval.align.parsing import split_text_into_tokens, parse_multivariant_string
from asr_eval.align.recursive import align


def test_edit_distance():
    rng = np.random.default_rng(0)
    
    for _ in range(100):
        true = [str(x) for x in rng.integers(low=0, high=3, size=rng.integers(low=0, high=15))]
        pred = [str(x) for x in rng.integers(low=0, high=3, size=rng.integers(low=0, high=15))]
        assert (
            align([Token(x) for x in true], [Token(x) for x in pred]).score.n_word_errors
            == nltk.edit_distance(true, pred) # pyright: ignore[reportUnknownMemberType]
        )


def test_align_recursive():
    true_text = 'a <*> b c {x|y|d} {qaz|} {a|b}'
    pred_text = 'a b x y a a'
    true = parse_multivariant_string(true_text)
    pred = split_text_into_tokens(pred_text)
    matches_list = align(true, pred)
    
    print([
        ([t.value for t in match.true], [t.value for t in match.pred])
        for match in matches_list.matches
    ])

    assert [
        ([t.value for t in match.true], [t.value for t in match.pred])
        for match in matches_list.matches
    ] == [
        (['a'], ['a']),
        (['<*>'], []),
        (['b'], ['b']),
        (['c'], ['x']),
        (['y'], ['y']),
        (['a'], ['a']),
        ([], ['a'])
    ]

    for x in true:
        if isinstance(x, Token):
            assert true_text[x.pos[0]:x.pos[1]] == x.value
        else:
            for option in x.options:
                for x2 in option:
                    assert true_text[x2.pos[0]:x2.pos[1]] == x2.value

    for x in pred:
        assert isinstance(x, Token)
        assert pred_text[x.pos[0]:x.pos[1]] == x.value