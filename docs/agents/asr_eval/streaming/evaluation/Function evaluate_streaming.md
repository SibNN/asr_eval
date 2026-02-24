# Function evaluate_streaming (defined in asr_eval/streaming/evaluation.py at lines 163-252)

def evaluate_streaming(
    timed_transcription: asr_eval.align.transcription.Transcription,
    waveform: asr_eval.utils.types.FLOATS,
    cutoffs: list[asr_eval.streaming.sender.Cutoff],
    input_chunks: list[asr_eval.streaming.model.InputChunk],
    output_chunks: list[asr_eval.streaming.model.OutputChunk],
    partial_alignment_interval: float = 0.25,
) -> asr_eval.streaming.evaluation.StreamingEvaluationResults:
    """An automation to evaluate streaming recognition results.

    Aligns partial transcriptions against starting parts of the ground
    truth.

    For each of the :code:`timestamps` obtains ths starting part of the
    :code:`timed_transcription` up to the specified timestamp, and
    aligns against the partial transcription that was received up to
    the specified timestamp. If the timestamp is inside a word in the
    ground truth transcription, considers two partial true
    transcriptions - with and without this word - and selects one with
    the best alignment score.

    Args:
        timed_transcription: The ground truth transcription for the
            whole audio with filled timings for each token. Is typically
            obtained with
            :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        waveform: A waveform in float32 dtype with sampling rate 16000.
            Note that the streaming recognizer may accept a different
            sampling rate or dtype. A conversion to the required rate
            and dtype is typically done on-the-fly inside
            :func:`~asr_eval.streaming.evaluation.make_sender`
            function.
        cutoffs: A schedule on which the input chunks was sent.
        input_chunks: The input chunks history. Will create a copy of
            each chunk with relative timestamps instead of absolute.
            Will not modify the original chunks.
        output_chunks: The outputs chunks history. Will create a copy of
            each chunk with relative timestamps instead of absolute.
            Will not modify the original chunks.
        partial_alignment_interval: Time interval between consecutive
            alignments of the partial transcriptions against starting
            parts of the ground truth. Is real-timescale: for example,
            if a 10 sec long audios is transcribed for 30 seconds, and
            :code:`partial_alignment_interval=1`, then we will get
            30 partial alignments.

    Returns:
        A :class:`~asr_eval.streaming.evalution.StreamingEvaluationResults`
        dataclass that scores the resulting partial alignments, as well
        as the input data.
    """
    ...