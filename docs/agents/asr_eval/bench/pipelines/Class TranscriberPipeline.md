# Class TranscriberPipeline (defined in asr_eval/bench/pipelines/_registry.py at lines 20-121)

class TranscriberPipeline(abc.ABC):
    """An abstract class for pipelines.

    Pipeline is any speech recognition algorithm that processes audio
    into text or timed text. Each pipeline is stored under unqiue name.

    See Also:
        More details and examples in the user guides
        :doc:`/guide_evaluation_dashboard`.

    See many examples in `asr_eval.bench.pipelines._registered` package.

    To register a pipeline, you need to subclass as follows:

    Example:
        >>> from datasets import load_dataset, Audio
        >>> from asr_eval.bench.pipelines import TranscriberPipeline, get_pipeline
        >>> from asr_eval.models.base.longform import LongformCTC
        >>> from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper
        >>> class _(TranscriberPipeline, register_as='example-wav2vec2'):
        ...     def init(self):
        ...         # override init to return a pipeline instance
        ...         return LongformCTC(
        ...             Wav2vec2Wrapper('facebook/wav2vec2-base-960h')
        ...         )

        >>> # now you can load the registered pipeline:
        >>> pipeline_instance = get_pipeline('example-wav2vec2')()
        >>> dataset = (
        ...     load_dataset('PolyAI/minds14', name='en-US', split='train')
        ...     .cast_column('audio', Audio(sampling_rate=16_000))
        ... )
        >>> sample = dataset[4]
        >>> pipeline_instance.run(sample)  # doctest: +SKIP
        {'text': 'CAN NOW YOU HELP ME SET UP AN JOINT LEAKACCOUNT ',
            'elapsed_time': 0.23598575592041016}
    """
    ...

    @abc.abstractmethod
    ...

    def run(self, sample: asr_eval.bench.datasets.AudioSample) -> dict[str, typing.Any]:
    ...