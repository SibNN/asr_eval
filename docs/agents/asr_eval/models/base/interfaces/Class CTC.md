# Class CTC (defined in asr_eval/models/base/interfaces.py at lines 68-122)

class CTC(asr_eval.models.base.interfaces.Transcriber):
    """An abstract CTC model that converts audio into log probabilties
    for each time frame.

    Implementations override a list of additional methods for additional
    actions such as decoding.
    """
    ...

    @abc.abstractmethod
    def ctc_log_probs(self, waveforms: list[asr_eval.utils.types.FLOATS]) -> list[asr_eval.utils.types.FLOATS]:
        """Calculates log probabilties each time frame, given a float32
        waveform, typically normalized from -1 to 1. Exponent from the
        log probabilties should sum up to 1 for each time frame.

        Typically obtained from logits via
        :code:`torch.nn.functional.log_softmax`. Note that the returned
        value should be a numpy array, not a torch tensor.
        """
        ...

    @property
    @abc.abstractmethod
    def blank_id(self) -> int:
        """An index in vocabulary for <blank> CTC token."""
        ...

    @property
    @abc.abstractmethod
    def tick_size(self) -> float:
        """A time interval in seconds between consecutive time frames in
        the log probs matrix.
        """
        ...

    @property
    @abc.abstractmethod
    def vocab(self) -> tuple[str, ...]:
        """Returns a vocabulary: a character (usually a single letter)
        or character sequence for each vocabulary index, or empty string
        for blank token.

        Note that this does not fully support Whisper-style BPE
        encoding: each single token should correspond to a valid unicode
        string.
        """
        ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
        # converts CTC log probs into the output text via argmax
        ...