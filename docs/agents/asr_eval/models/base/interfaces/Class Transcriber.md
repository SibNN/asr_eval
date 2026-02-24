# Class Transcriber (defined in asr_eval/models/base/interfaces.py at lines 33-43)

class Transcriber(abc.ABC):
    """An abstract transcriber (audio -> text) to evaluate on any
    dataset.
    """
    ...

    @abc.abstractmethod
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1.
        """
        ...