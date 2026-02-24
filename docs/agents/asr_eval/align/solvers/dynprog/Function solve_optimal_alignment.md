# Function solve_optimal_alignment (defined in asr_eval/align/solvers/dynprog.py at lines 48-282)

def solve_optimal_alignment(
    true: asr_eval.align.transcription.Transcription, pred: asr_eval.align.transcription.Transcription
) -> tuple[asr_eval.align.matching.MatchesList, list[int]]:
    """
    Solves an optimal alignment task via dynamic programming. Uses a
    generalized version of the Needleman-Wunsch algorithm with the
    following modifications:

    1. Support for multivariant blocks in both texts.
    2. Support for :class:`~asr_eval.align.transcription.Wildcard`
       symbols in both texts.
    3. Better alignment due to the optimization of additional metrics
       (see the :class:`~asr_eval.align.matching.AlignmentScore` for
       details).

    The second returned value contains a selected option index for
    each multivariant block in `true`.

    Note:
        In the *asr_eval* workflow `pred` is single-variant. However,
        the algorithm supports multivariant blocks and
        :class:`~asr_eval.align.transcription.Wildcard` symbols for
        both `true` and `pred`.

    Example:
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.solvers.dynprog import solve_optimal_alignment
        >>> true = 'hey <*> {eh} {one|1} {dollar|$}'
        >>> pred = 'Hey man eh dollar'
        >>> matches_list, selected_blocks = solve_optimal_alignment(
        ...     DEFAULT_PARSER.parse_transcription(true),
        ...     DEFAULT_PARSER.parse_transcription(pred),
        ... )
        >>> matches_list.score
        AlignmentScore(n_word_errors=1, n_correct=3, n_char_errors=1)
        >>> # selected #0 in {eh|}, #1 in {one|1}, #0 in {dollar|$}
        >>> selected_blocks
        [0, 1, 0]

    """
    ...