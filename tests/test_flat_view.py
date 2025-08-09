from asr_eval.align.transcription import TOKEN_UID, FlatLoc
from asr_eval.align.parsing import parse_multivariant_string


def test_flat_view():
    true = parse_multivariant_string(
        '{седьмого|7} - {восьмого|8} мая {|в} {Пуэрто-Рико} прошёл {шестнадцатый|16-й|16й|16} этап'
    )
    view = true.flat_view()

    uid_to_value = {
        token.uid: token.value
        for token in true.list_all_tokens()
    }

    def _resolve(uid_or_special: TOKEN_UID | FlatLoc) -> str:
        nonlocal uid_to_value
        if isinstance(uid_or_special, str):
            return str(uid_to_value[uid_or_special])
        return str(uid_or_special.name)

    for from_idx, to_idxs in enumerate(view.transitions):
        print('from', _resolve(view.positions[from_idx]), 'to', [_resolve(view.positions[i]) for i in to_idxs])
    
    results = [
        (
            'from',
            _resolve(view.positions[from_idx]),
            'to',
            [_resolve(view.positions[i]) for i in to_idxs]
        )
        for from_idx, to_idxs in enumerate(view.transitions)
    ]
    
    assert results == [
        ('from', 'Start', 'to', ['седьмого', '7']),
        ('from', 'седьмого', 'to', ['восьмого', '8']),
        ('from', '7', 'to', ['восьмого', '8']),
        ('from', 'восьмого', 'to', ['мая']),
        ('from', '8', 'to', ['мая']),
        ('from', 'мая', 'to', ['в', 'пуэрто', 'прошел']),
        ('from', 'в', 'to', ['пуэрто', 'прошел']),
        ('from', 'пуэрто', 'to', ['рико']),
        ('from', 'рико', 'to', ['прошел']),
        ('from', 'прошел', 'to', ['шестнадцатый', '16', '16й', '16']),
        ('from', 'шестнадцатый', 'to', ['этап']),
        ('from', '16', 'to', ['й']),
        ('from', 'й', 'to', ['этап']),
        ('from', '16й', 'to', ['этап']),
        ('from', '16', 'to', ['этап']),
        ('from', 'этап', 'to', ['End']),
    ]