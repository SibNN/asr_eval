# Function forced_alignment (defined in asr_eval/ctc/forced_alignment.py at lines 23-98)

def forced_alignment(
    log_probs: asr_eval.utils.types.FLOATS,
    true_tokens: list[int] | asr_eval.utils.types.INTS,
    blank_id: int = 0,
) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    """Performs a forced alignment.

    Returns the path with the highest cumulative probability among all
    paths that match the specified transcription.

    Args:
        log_probs: log probabilities from CTC model.
        true_tokens: a sequence of tokens for the ground truth
            transcription.
        blank_id: an index for :code:`<blank>` CTC token.

    Returns:
        A tuple. The first element is the token for each frame. The
        second element is a probability for each frame. The third
        element is a frame span (start_position, end_position) for
        each of true_tokens.

    Note:
        This is going to stop working for torchaudio>=2.9.0, see
        https://github.com/pytorch/audio/issues/3902 . It is possible
        to use :func:`~asr_eval.ctc.forced_alignment.recursion_forced_alignment`,
        but there may be problems with recursion limit (recursion limit
        is 1000 for Python, equals 20 sec with 50 ticks/sec). To use
        custom implementation, set environmental variable
        :code:`FORCED_ALIGN_CUSTOM=1`.


    """
    ...