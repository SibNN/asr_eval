from asr_eval.align.data import Token
from asr_eval.align.parsing import parse_string, parse_multi_variant_string
from asr_eval.align.recursive import align


def test_align_recursive():
    true_text = 'a <*> b c {x|y|d} {qaz|} {a|b}'
    pred_text = 'a b x y a a'
    true = parse_multi_variant_string(true_text)
    pred = parse_string(pred_text)
    matches_list = align(true, pred)

    assert [
        ([t.value for t in match.true], [t.value for t in match.pred])
        for match in matches_list.matches
    ] == [
        (['a'], ['a']),
        (['<*>'], []),
        (['b'], ['b']),
        (['c'], ['x']),
        (['y'], ['y']),
        (['qaz'], ['a']),
        (['a'], ['a'])
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