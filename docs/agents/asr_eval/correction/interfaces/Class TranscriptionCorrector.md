# Class TranscriptionCorrector (defined in asr_eval/correction/interfaces.py at lines 11-20)

class TranscriptionCorrector(abc.ABC):
    """An abstract postprocessor capable of correcting ASR
    transcriptions.
    """
    ...

    @abc.abstractmethod
    def correct(
        self, transcription: str, waveform: asr_eval.utils.types.FLOATS | None = None
        ...