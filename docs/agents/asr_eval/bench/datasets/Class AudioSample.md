# Class AudioSample (defined in asr_eval/bench/datasets/_registry.py at lines 49-104)

class AudioSample(typing.TypedDict):
    """A TypedDict typization for Hugging Face audio sample in
    standard asr_eval format.

    This class is for typing purposes only. A sample in Hugging Face
    dataset is a plain dict.

    In asr_eval standard workflow, sampling rate should be 16_000 and
    all samples should have unique "sample_id" value. Dataset may
    include other custom fields as well.

    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.

    Example:

        >>> # instantiation from `get_dataset`:
        >>> from asr_eval.bench.datasets import get_dataset
        >>> dataset = get_dataset('podlodka')
        >>> sample: AudioSample = dataset[0]

        >>> # AudioSample inner structure:
        >>> audio_data: AudioData = sample['audio']
        >>> assert audio_data['sampling_rate'] == 16_000
        >>> waveform: FLOATS = audio_data['array']
        >>> transcription: str = sample['transcription']

        >>> # instantiation from Hugging Face:
        >>> from datasets import load_dataset, Audio
        >>> from asr_eval.bench.datasets import AudioSample, AudioData
        >>> from asr_eval.utils.types import FLOATS
        >>> from asr_eval.bench.datasets.mappers import assign_sample_ids
        >>> dataset = (
        ...     load_dataset('PolyAI/minds14', name='en-US', split='train')
        ...     .cast_column('audio', Audio(sampling_rate=16_000))
        ...     .map(assign_sample_ids, with_indices=True)
        ... )
        >>> sample: AudioSample = dataset[0]
    """
    ...

    audio: asr_eval.bench.datasets._registry.AudioData
    """An `Audio <https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature>`_
    feature. In asr_eval standard workflow, should be obtained with
    :code:`.cast_column('audio', Audio(decode=True, sampling_rate=16_000))`.
    """

    transcription: str
    """A transcription as text, possibly with multivariant annotation,
    may optionally include punctuation or capitalization.
    """

    sample_id: int
    """A sample ID that should be unique in the dataset. Normally should
    equal a sample index in the unshuffled and not filtered version."""