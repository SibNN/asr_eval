# Class AudioSpectrogramTransformer (defined in asr_eval/models/ast_wrapper.py at lines 15-105)

class AudioSpectrogramTransformer:
    """An AudioSpectrogramTransformer (AST) able to recognize sound
    types.

    Requires :code:`transformers` package.
    """
    ...

    def predict_on_batch(
        self, waveforms: asr_eval.utils.types.FLOATS, sampling_rate: int = 16_000
    ) -> asr_eval.utils.types.FLOATS:   
    ...

    def predict_longform(
        self,
        waveform: asr_eval.utils.types.FLOATS,
        batch_size: int = 32,
        segment_length: float = 10,  # a train-time value for AST
        segment_shift: float = 5,
        sampling_rate: int = 16_000,
        # if <min_length, don't want to predict, too short and considered OOD
        min_length: float = 1,
    ) -> asr_eval.utils.types.FLOATS:
    ...

    def plot_top_classes(
        self, logits: asr_eval.utils.types.FLOATS, top_by: typing.Literal['max', 'mean'] = 'max'
    ):
        # logits have shape (n_segments, n_classes)
        ...