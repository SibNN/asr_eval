# Class Segmenter (defined in asr_eval/models/base/interfaces.py at lines 18-31)

class Segmenter(abc.ABC):
    """An abstract model that segments a long-form audio into chunks
    containing speech.

    Any parameters, such as max segment size, should go into a class
    constructor.
    """
    ...