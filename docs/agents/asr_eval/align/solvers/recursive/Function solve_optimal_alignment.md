# Function solve_optimal_alignment (defined in asr_eval/align/solvers/recursive.py at lines 23-183)

def solve_optimal_alignment(
    true: asr_eval.align.transcription.Transcription,
    pred: asr_eval.align.transcription.SingleVariantTranscription,
    determine_selected_multivariant_indices: bool = True,
) -> tuple[asr_eval.align.matching.MatchesList, list[int]]:
    """Solves an optimal alignment task via recursion. Supports
    multivariant annotations with
    :class:`~asr_eval.align.transcription.Wildcard` insertions.

    Note:
        This method is legacy, consired using
        :func:`asr_eval.align.solvers.dynprog.solve_optimal_alignment`
        instead.

    The last returned value is a selected option index for each
    multivariant block, if present in the ground truth.
    """
    ...