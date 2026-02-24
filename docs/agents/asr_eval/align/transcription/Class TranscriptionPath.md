# Class TranscriptionPath (defined in asr_eval/align/transcription.py at lines 498-754)

@dataclasses.dataclass(frozen=True)
class TranscriptionPath(asr_eval.align.transcription.Transcription):
    """ A Transcription with a selected option for each multivariant
    block.

    Note:
        This is a lower-level subclass that extends
        :class:`~asr_eval.align.transcription.Transcription` with a few
        indexing methods that are typically not called manually. See the
        :class:`~asr_eval.align.transcription.Transcription`
        docs for the main methods.

    See more details in :doc:`/guide_alignment_wer`.
    """
    ...

    multivariant_choices: tuple[int, ...]

    def get_prev_slot(
        self, loc: asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc
    ) -> asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc | None:
        """A step backward: from the next slot to the previous.

        Returns None if we reached the end.
        """
        ...

    def get_next_slot(
        self, loc: asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc
    ) -> asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc | None:
        """A step forward: from the previous slot to the next.

        Returns None if we reached the beginning.
        """
        ...

    def slot_idx_to_loc(self, index: int) -> asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc:
        """Get a slot by slot index."""
        ...

    def slot_loc_to_idx(self, loc: asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc) -> int:
        """Get a slot index for a given slot."""
        ...

    def token_uid_to_slot(
        self, uid: asr_eval.align.transcription.TOKEN_UID
    ) -> tuple[int, asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc]:
        """Get a slot and a slot index for a given Token
        :attr:`~asr_eval.align.transcription.Token.uid`.
        """
        ...

    def slot_to_token(self, loc: asr_eval.align.transcription.InnerLoc | asr_eval.align.transcription.OuterLoc) -> asr_eval.align.transcription.Token:
        """Get a Token for a given "at" index (outer or inner)."""
        ...