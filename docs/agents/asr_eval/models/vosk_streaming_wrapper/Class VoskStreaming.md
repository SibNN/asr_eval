# Class VoskStreaming (defined in asr_eval/models/vosk_streaming_wrapper.py at lines 21-125)

class VoskStreaming(asr_eval.streaming.model.StreamingASR):
    """A wrapper for Vosk streaming model.

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @property
    @typing.override
    def audio_type(self) -> typing.Literal['bytes']:
    ...