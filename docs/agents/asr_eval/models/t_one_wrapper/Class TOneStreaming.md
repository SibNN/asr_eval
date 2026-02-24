# Class TOneStreaming (defined in asr_eval/models/t_one_wrapper.py at lines 23-78)

class TOneStreaming(asr_eval.streaming.model.StreamingASR):
    """A streaming wrapper for T-One model.

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @property
    @typing.override
    def audio_type(self) -> typing.Literal['int']:
    ...