# Function get_outer_slots_values (defined in asr_eval/align/transcription.py at lines 484-496)

def get_outer_slots_values(
    blocks: collections.abc.Sequence[asr_eval.align.transcription.Token | asr_eval.align.transcription.MultiVariantBlock]
) -> collections.abc.Iterator[asr_eval.align.transcription.Token | asr_eval.align.transcription.MultiVariantBlock | None]:
    """Enumerates all the outer slot values in the ground truth.

    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`.
    """
    ...