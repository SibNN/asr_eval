# Class SaluteWrapper (defined in asr_eval/models/salute_wrapper.py at lines 16-68)

class SaluteWrapper(asr_eval.models.base.interfaces.TimedTranscriber):
    """ A wrapper for SaluteSpeech API transcriber.

    Need to pass api_key:
    https://developers.sber.ru/docs/ru/salutespeech/quick-start/integration-individuals

    Raises:
        salute_speech.exceptions.SberSpeechError: on API errors

    Installation: see :doc:`/guide_installation` page.
    """
    ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...