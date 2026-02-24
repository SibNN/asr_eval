# Class DummyASR (defined in asr_eval/streaming/model.py at lines 597-640)

class DummyASR(asr_eval.streaming.model.StreamingASR):
    """Will transcribe N seconds long audio into "1 2 ... N"."""
    ...

    @property
    @typing.override
    def audio_type(self) -> typing.Literal['float', 'int', 'bytes']:
    ...