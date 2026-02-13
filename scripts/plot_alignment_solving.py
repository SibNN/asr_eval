import matplotlib.pyplot as plt
import matplotlib.patches as patches

from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.solvers.dynprog import _FlatLoc, _flat_view # pyright: ignore[reportPrivateUsage]


true_text = '{uh} please {give me|gimme} that {1|one} {eh}'
from asr_eval.align.parsing import DEFAULT_PARSER
true = DEFAULT_PARSER.parse_transcription(true_text)
true_view = _flat_view(true)
true_tokens = {token.uid: token for token in true.list_all_tokens()}
true_finish_mask = [len(true_view.transitions) in t for t in true_view.transitions] + [True]

pred_text = 'oh please give that one'
pred = DEFAULT_PARSER.parse_transcription(pred_text)
pred_view = _flat_view(pred)
pred_tokens = {token.uid: token for token in pred.list_all_tokens()}
pred_finish_mask = [len(pred_view.transitions) in t for t in pred_view.transitions] + [True]

fix, ax = plt.subplots(figsize=(14, 5)) # type: ignore

for pos_idx, token_uid in enumerate(true_view.positions):
    match token_uid:
        case _FlatLoc.Start:
            text = '<start>'
        case _FlatLoc.End:
            text = '<end>'
        case _:
            text = str(true_tokens[token_uid].value)
    plt.text( # type: ignore
        -1,
        pos_idx,
        text,
        ha='right',
        va='center',
        fontsize=12,
        color='green' if true_finish_mask[pos_idx] else 'black',
    )
    if token_uid != _FlatLoc.End:
        plt.text( # type: ignore
            -0.5,
            pos_idx,
            f'({pos_idx})',
            ha='right',
            va='center',
            fontsize=9,
            color='green' if true_finish_mask[pos_idx] else 'black',
        )

for from_idx, to_idxs in enumerate(true_view.transitions):
    for to_idx in to_idxs:
        ax.add_patch(patches.FancyArrowPatch(
            (-3, from_idx),
            (-3, to_idx),
            connectionstyle='arc3, rad=-0.5',
            arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
            color='green' if true_finish_mask[from_idx] else 'black',
        ))

for pos_idx, token_uid in enumerate(pred_view.positions):
    match token_uid:
        case _FlatLoc.Start:
            text = '<start>'
        case _FlatLoc.End:
            text = '<end>'
        case _:
            text = str(pred_tokens[token_uid].value)
    plt.text( # type: ignore
        pos_idx,
        -1,
        text,
        rotation=90,
        va='top',
        ha='center',
        fontsize=12,
        color='green' if pred_finish_mask[pos_idx] else 'black',
    )
    if token_uid != _FlatLoc.End:
        plt.text( # type: ignore
            pos_idx,
            -0.5,
            f'({pos_idx})',
            va='top',
            ha='center',
            fontsize=9,
            color='green' if pred_finish_mask[pos_idx] else 'black',
        )

for from_idx, to_idxs in enumerate(pred_view.transitions):
    for to_idx in to_idxs:
        ax.add_patch(patches.FancyArrowPatch(
            (from_idx, -3),
            (to_idx, -3),
            connectionstyle='arc3, rad=0.5',
            arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
            color='green' if pred_finish_mask[from_idx] else 'black',
        ))

for true_idx in range(len(true_view.transitions)):
    for pred_idx in range(len(pred_view.transitions)):
        if true_finish_mask[true_idx] and pred_finish_mask[pred_idx]:
            color = 'green'
        else:
            color = 'lightblue'
        plt.scatter([pred_idx], [true_idx], s=50, color=color) # type: ignore

true_idx = 2
pred_idx = 1

for true_to_idx in [true_idx] + true_view.transitions[true_idx]:
    for pred_to_idx in [pred_idx] + pred_view.transitions[pred_idx]:
        if true_to_idx == true_idx and pred_to_idx == pred_idx:
            continue
        if true_to_idx > true_idx and pred_to_idx > pred_idx:
            color = 'cyan'
        elif true_to_idx > true_idx:
            color = 'orange'
        else:
            color = 'magenta'
        ax.add_patch(patches.FancyArrowPatch(
            (pred_idx, true_idx),
            (pred_to_idx, true_to_idx),
            connectionstyle='arc3, rad=-0.15',
            arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
            color=color,
        ))

legend_x = len(pred_tokens) + 2
legend_y = len(true_tokens)
legend_dy = 0.8

plt.text(legend_x, legend_y, f'true:', va='center', ha='left', fontsize=10, color='black') # type: ignore
plt.text(legend_x, legend_y - legend_dy, true_text, va='center', ha='left', fontsize=10, color='black') # type: ignore

plt.text(legend_x, legend_y - 2.5 * legend_dy, f'pred:', va='center', ha='left', fontsize=10, color='black') # type: ignore
plt.text(legend_x, legend_y - 3.5 * legend_dy, pred_text, va='center', ha='left', fontsize=10, color='black') # type: ignore

plt.scatter(legend_x, legend_y - 5.2 * legend_dy, s=50, color='lightblue') # type: ignore
plt.text(legend_x + 0.4, legend_y - 5.25 * legend_dy, 'non-finish positions', va='center', ha='left') # type: ignore

plt.scatter(legend_x, legend_y - 6.45 * legend_dy, s=50, color='green') # type: ignore
plt.text(legend_x + 0.4, legend_y - 6.5 * legend_dy, 'finish positions', va='center', ha='left') # type: ignore

for delta, color, name in [
    (8, 'orange', 'deletion transitions'),
    (9.2, 'cyan', 'replacement/correct transitions'),
    (10.4, 'magenta', 'insertion transitions'),
]:
    ax.add_patch(patches.FancyArrowPatch(
        (legend_x, legend_y - delta * legend_dy),
        (legend_x + 1.5, legend_y - delta * legend_dy),
        connectionstyle='arc3, rad=-0.0',
        arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
        color=color,
    ))
    plt.text(legend_x + 1.9, legend_y - delta * legend_dy, name, va='center', ha='left') # type: ignore

plt.xlim(-4, len(pred_tokens) + 15) # type: ignore
plt.ylim(-3, len(true_tokens) + 1) # type: ignore
plt.axis('off') # type: ignore
plt.savefig('pub/alignment_solving.svg') # type: ignore
plt.savefig('docs/source/images/alignment/plot.png') # type: ignore
plt.close()