# Class Qwen2AudioWrapper (defined in asr_eval/models/qwen2_audio_wrapper.py at lines 12-92)

class Qwen2AudioWrapper(asr_eval.models.base.interfaces.ContextualTranscriber):
    '''A wrapper for  Qwen2-Audio transcriber.

    Produces bad output, TODO fix.

    If domain_text is specified, it is added into prompt with a note
    "may be related".

    Installation: see :doc:`/guide_installation` page.

    Authors: Muharyam Baviev & Oleg Sedukhin
    '''
    ...

    @typing.override
    def contextual_transcribe(
        self, waveform: asr_eval.utils.types.FLOATS, prev_transcription: str = ''
    ) -> str:
    ...