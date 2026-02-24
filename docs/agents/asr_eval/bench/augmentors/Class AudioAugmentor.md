# Class AudioAugmentor (defined in asr_eval/bench/augmentors/_registry.py at lines 15-36)

class AudioAugmentor(abc.ABC):
    """Abstract audio preprocessor, primarily for evaluation with
    artificial noises.

    To register an augmentor, one need to subclass this class and define
    the :code:`__call__` method that processes an audio sample.

    Preferrably should not modify the input dict and return a copy.

    TODO example.
    """
    ...