from asr_eval.align.alignment import Alignment, MultipleAlignment
from asr_eval.align.parsing import DEFAULT_PARSER

def test_multiple_alignment():
    
    true_text = 'a <*> b {c|d} {e} f'
    pred_text = 'a b f'

    true = DEFAULT_PARSER.parse_transcription(true_text)
    pred = DEFAULT_PARSER.parse_single_variant_transcription(pred_text)
    al = Alignment(true, pred)
    multiple_alignment = MultipleAlignment(
        baseline=true,
        alignments={'pred': al},
    )
    multiple_alignment.render_as_text()