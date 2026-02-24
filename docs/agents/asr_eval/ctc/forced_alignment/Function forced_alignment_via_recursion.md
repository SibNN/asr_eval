# Function forced_alignment_via_recursion (defined in asr_eval/ctc/forced_alignment.py at lines 100-170)

def forced_alignment_via_recursion(
    log_probs: asr_eval.utils.types.FLOATS,
    true_tokens: list[int] | asr_eval.utils.types.INTS,
    blank_id: int = 0,
) -> tuple[list[int], list[float]]:
    """Performs forced alignment via a custom recursive algorithm."""
    ...