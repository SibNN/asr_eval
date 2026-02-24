# Function get_partial_alignments (defined in asr_eval/streaming/evaluation.py at lines 461-565)

def get_partial_alignments(
    input_history: typing.Sequence[asr_eval.streaming.model.InputChunk],
    output_history: typing.Sequence[asr_eval.streaming.model.OutputChunk],
    timed_transcription: asr_eval.align.transcription.Transcription,
    timestamps: list[float] | asr_eval.utils.types.FLOATS | None = None,
    processes: int = 1,
) -> list[asr_eval.streaming.evaluation.PartialAlignment]:
    """Aligns partial transcriptions against starting parts of the
    ground truth.

    For each of the :code:`timestamps` obtains ths starting part of the
    :code:`timed_transcription` up to the specified timestamp, and
    aligns against the partial transcription that was received up to
    the specified timestamp. If the timestamp is inside a word in the
    ground truth transcription, considers two partial true
    transcriptions - with and without this word - and selects one with
    the best alignment score.

    Args:
        input_history: The input chunks history.
        output_history: The output chunks history.
        true_word_timings: The ground truth transcription for the
            whole audio with filled timings for each token. Is typically
            obtained with
            :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        timestamps: A list of times when to evaluate partial results. If
            None, will evaluate after each of the output chunks, except
            the last :code:`Signal.FINISH` chunk if present.
        processes: If > 1, paralellizes using multiprocessing (we cannot
            use multithreading here because of GIL, considering that the
            alignment function is written on pure Python).
    """
    ...