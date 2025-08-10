from dataclasses import asdict
from asr_eval.align.solver import FlatLoc, flat_view
from asr_eval.align.parsing import parse_multivariant_string
from asr_eval.align.transcription import TOKEN_UID, MultiVariantBlock


def test_flat_view_short():
    view = flat_view(parse_multivariant_string('{a}'))
    assert asdict(view) == {
        'positions': [FlatLoc.Start, 'id0', FlatLoc.End],
        'transitions': [[1, 2], [2]],
        'resolved_multivariant_blocks': {
            (0, 1): [('mvid0', 0)],
            (0, 2): [('mvid0', 1)],
        },
    }

    view = flat_view(parse_multivariant_string('{|a}'))
    assert asdict(view) == {
        'positions': [FlatLoc.Start, 'id0', FlatLoc.End],
        'transitions': [[1, 2], [2]],
        'resolved_multivariant_blocks': {
            (0, 1): [('mvid0', 1)],
            (0, 2): [('mvid0', 0)],
        },
    }

    view = flat_view(parse_multivariant_string('a {b c} {c}'))
    assert asdict(view) == {
        'positions': [FlatLoc.Start, 'id0', 'id1', 'id2', 'id3', FlatLoc.End],
        'transitions': [[1], [2, 4, 5], [3], [4, 4, 5], [5]],
        'resolved_multivariant_blocks': {
            (1, 2): [('mvid0', 0)],
            (1, 4): [('mvid0', 1), ('mvid1', 0)],
            (3, 4): [('mvid1', 0)],
            (1, 5): [('mvid0', 1), ('mvid1', 1)],
            (3, 5): [('mvid1', 1)],
        },
    }


def test_flat_view_long():
    true = parse_multivariant_string(
        '{седьмого|7} - {восьмого|8} мая {|в} {Пуэрто-Рико} прошёл {шестнадцатый|16-й|16й|16} этап'
    )
    view = flat_view(true)

    uid_to_value = {
        token.uid: token.value
        for token in true.list_all_tokens()
    }

    def to_str(elem: TOKEN_UID | FlatLoc) -> str:
        nonlocal uid_to_value
        if isinstance(elem, str):
            return str(uid_to_value[elem])
        return str(elem)

    def mvar_block_to_str(uid: str) -> str:
        nonlocal true
        for block in true.tokens:
            if isinstance(block, MultiVariantBlock) and block.uid == uid:
                return block.to_text()
        assert False

    verbose_transitions = []

    for from_idx, to_idxs in enumerate(view.transitions):
        for to_idx in to_idxs:
            resolved_blocks = view.resolved_multivariant_blocks.get((from_idx, to_idx), [])
            resolved_repr = [
                (mvar_block_to_str(resolved_block[0]), resolved_block[1])
                for resolved_block in resolved_blocks
            ]
            verbose_transitions.append(( # type: ignore
                'from',
                to_str(view.positions[from_idx]),
                'to',
                to_str(view.positions[to_idx]),
                'resolved',
                *resolved_repr,
            ))
        
    assert verbose_transitions == [
        ('from', 'FlatLoc.Start', 'to', 'седьмого', 'resolved', ('{седьмого|7}', 0)),
        ('from', 'FlatLoc.Start', 'to', '7', 'resolved', ('{седьмого|7}', 1)),
        ('from', 'седьмого', 'to', 'восьмого', 'resolved', ('{восьмого|8}', 0)),
        ('from', 'седьмого', 'to', '8', 'resolved', ('{восьмого|8}', 1)),
        ('from', '7', 'to', 'восьмого', 'resolved', ('{восьмого|8}', 0)),
        ('from', '7', 'to', 'восьмого', 'resolved', ('{восьмого|8}', 0)),
        ('from', '7', 'to', '8', 'resolved', ('{восьмого|8}', 1)),
        ('from', 'восьмого', 'to', 'мая', 'resolved'),
        ('from', '8', 'to', 'мая', 'resolved'),
        ('from', 'мая', 'to', 'в', 'resolved', ('{|в}', 1)),
        ('from', 'мая', 'to', 'пуэрто', 'resolved', ('{|в}', 0), ('{пуэрто рико|}', 0)),
        ('from', 'мая', 'to', 'прошел', 'resolved', ('{|в}', 0), ('{пуэрто рико|}', 1)),
        ('from', 'в', 'to', 'пуэрто', 'resolved', ('{пуэрто рико|}', 0)),
        ('from', 'в', 'to', 'пуэрто', 'resolved', ('{пуэрто рико|}', 0)),
        ('from', 'в', 'to', 'прошел', 'resolved', ('{пуэрто рико|}', 1)),
        ('from', 'пуэрто', 'to', 'рико', 'resolved'),
        ('from', 'рико', 'to', 'прошел', 'resolved'),
        ('from', 'прошел', 'to', 'шестнадцатый', 'resolved', ('{шестнадцатый|16 й|16й|16}', 0)),
        ('from', 'прошел', 'to', '16', 'resolved', ('{шестнадцатый|16 й|16й|16}', 1)),
        ('from', 'прошел', 'to', '16й', 'resolved', ('{шестнадцатый|16 й|16й|16}', 2)),
        ('from', 'прошел', 'to', '16', 'resolved', ('{шестнадцатый|16 й|16й|16}', 3)),
        ('from', 'шестнадцатый', 'to', 'этап', 'resolved'),
        ('from', '16', 'to', 'й', 'resolved'),
        ('from', 'й', 'to', 'этап', 'resolved'),
        ('from', '16й', 'to', 'этап', 'resolved'),
        ('from', '16', 'to', 'этап', 'resolved'),
        ('from', 'этап', 'to', 'FlatLoc.End', 'resolved'),
    ]