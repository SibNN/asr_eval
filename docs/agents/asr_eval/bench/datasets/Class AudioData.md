# Class AudioData (defined in asr_eval/bench/datasets/_registry.py at lines 25-46)

class AudioData(typing.TypedDict):
    """A TypedDict typization for
    `Audio <https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature>`_
    feature in Hugging Face dataset.

    See examples in the docs for
    :class:`~asr_eval.bench.datasets.AudioSample`.
    """
    ...

    array: asr_eval.utils.types.FLOATS
    """1-D audio waveform of floats, normalized roughly from -1 to 1,
    with sampling rate specified in
    :attr:`~asr_eval.bench.datasets.AudioData.sampling_rate` (normally
    16000).
    """

    sampling_rate: int
    """A sampling rate for
    :attr:`~asr_eval.bench.datasets.AudioData.array`, i. e. array size
    per second (normally 16000).
    """