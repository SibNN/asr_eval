# Class CorrectorLangchain (defined in asr_eval/correction/corrector_langchain.py at lines 35-128)

class CorrectorLangchain(asr_eval.correction.interfaces.TranscriptionCorrector):
    """An agent that corrects a transcription, optionally with
    DuckDuckGo search.

    Works for Russian language currently.

    Requires :code:`langchain_openai` and :code:`duckduckgo_search`
    packages currently.

    Author: Timur Rafikov; Updated by: Oleg Sedukhin
    """
    ...

    @typing.override
    def correct(
        self, transcription: str, waveform: asr_eval.utils.types.FLOATS | None = None
    ) -> str:
    ...