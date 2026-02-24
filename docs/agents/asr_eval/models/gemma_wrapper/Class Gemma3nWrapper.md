# Class Gemma3nWrapper (defined in asr_eval/models/gemma_wrapper.py at lines 12-133)

class Gemma3nWrapper(asr_eval.models.base.interfaces.ContextualTranscriber):
    '''
    Gemma3n transcriber. Too slow currently, TODO fix

    If domain_text is specified, it is added into prompt with a note
    "may be related".

    Installation: see :doc:`/guide_installation` page.

    Authors: Timur Rafikov & Oleg Sedukhin
    '''
    ...

    @typing.override
    def contextual_transcribe(
        self, waveform: asr_eval.utils.types.FLOATS, prev_transcription: str = ''
    ) -> str:
        # processor calls self.feature_extractor(audio, ...), it
        # trims audio to 30 seconds
        ...