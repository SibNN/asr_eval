# Function draw_timed_transcription (defined in asr_eval/align/plots.py at lines 19-163)

def draw_timed_transcription(
    transcription: asr_eval.align.transcription.Transcription,
    y_pos: float = 0,
    y_delta: float = -1,
    y_tick_width: float = 0.1,
    ax: plt.Axes | None = None,
    graybox_y: tuple[float, float] | None = None,
):
    '''An utility to draw a transcription, possibly multivariant, with
    filled timings (see the full example in
    :func:`~asr_eval.align.timings.fill_word_timings_inplace`).

    Is used in streaming evaluation diagrams.
    '''
    ...