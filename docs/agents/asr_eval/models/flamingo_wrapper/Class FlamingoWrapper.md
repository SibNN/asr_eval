# Class FlamingoWrapper (defined in asr_eval/models/flamingo_wrapper.py at lines 14-61)

class FlamingoWrapper(asr_eval.models.base.interfaces.Transcriber):
    '''
    A Flamingo transcriber. Not working anymore, TODO fix

    Installation: see :doc:`/guide_installation` page.

    Authors: Dmitry Ezhov & Oleg Sedukhin
    '''
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
        # processor calls self.feature_extractor(audio, ...), it trims audio to 30 seconds
        ...