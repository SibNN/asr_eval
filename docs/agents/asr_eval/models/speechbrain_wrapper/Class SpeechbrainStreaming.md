# Class SpeechbrainStreaming (defined in asr_eval/models/speechbrain_wrapper.py at lines 21-109)

class SpeechbrainStreaming(asr_eval.streaming.model.StreamingASR):
    """
    A speechbrain streaming model asr-streaming-conformer-gigaspeech.

    Adopted from Gradio example from here:
    https://huggingface.co/speechbrain/asr-streaming-conformer-librispeech

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    def get_model(
        self, model_name: str
    ) -> speechbrain.inference.ASR.StreamingASR:
    ...

    @property
    @typing.override
    def audio_type(self) -> typing.Literal['float']:
    ...