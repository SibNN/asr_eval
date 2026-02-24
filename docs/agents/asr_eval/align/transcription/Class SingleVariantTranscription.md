# Class SingleVariantTranscription (defined in asr_eval/align/transcription.py at lines 435-444)

@dataclasses.dataclass(frozen=True)
class SingleVariantTranscription(asr_eval.align.transcription.Transcription):
    """A subclass of
    :class:`~asr_eval.align.transcription.Transcription` used
    for typing clarity where we do not expect multivariance (mainly for
    predictions). Typically constructed via
    :meth:`~asr_eval.align.parsers.DEFAULT_PARSER.parse_single_variant_transcription`.
    """
    ...

    blocks: tuple[asr_eval.align.transcription.Token, ...]