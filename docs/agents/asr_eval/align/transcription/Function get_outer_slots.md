# Function get_outer_slots (defined in asr_eval/align/transcription.py at lines 470-482)

def get_outer_slots(
    blocks: collections.abc.Sequence[asr_eval.align.transcription.Token | asr_eval.align.transcription.MultiVariantBlock]
) -> collections.abc.Iterator[asr_eval.align.transcription.OuterLoc]:
    """Enumerates all the outer slots in the ground truth.

    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    ...