from asr_eval.align.alignment import Alignment, MultipleAlignment
from asr_eval.align.matching import solve_optimal_alignment
from asr_eval.align.parsing import parse_multivariant_string, parse_single_variant_string

def test_multiple_alignment():
    
    true_text = 'a <*> b {c|d} {e} f'
    pred_text = 'a b f'

    true = parse_multivariant_string(true_text)
    pred = parse_single_variant_string(pred_text)
    matches_list, mv_indices = solve_optimal_alignment(true.tokens, pred.tokens)
    true_path = true.select_single_path(mv_indices)
    al = Alignment.from_matches(true_path, matches_list.matches)
    mal = MultipleAlignment(baseline=true, alignments={'pred': al})
    mal.view().render_as_text()