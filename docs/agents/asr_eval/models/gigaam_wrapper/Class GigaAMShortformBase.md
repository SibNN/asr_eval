# Class GigaAMShortformBase (defined in asr_eval/models/gigaam_wrapper.py at lines 26-69)

class GigaAMShortformBase(asr_eval.models.base.interfaces.Transcriber, abc.ABC):
    '''
    An abstract class for GigaAM model, either CTC or RNNT.

    Implementations:
        - :class:`~asr_eval.models.gigaam_wrapper.GigaAMShortformCTC`
        - :class:`~asr_eval.models.gigaam_wrapper.GigaAMShortformRNNT`
    '''
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...