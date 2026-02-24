# Class LongformVAD (defined in asr_eval/models/base/longform.py at lines 21-95)

class LongformVAD(asr_eval.models.base.interfaces.TimedTranscriber):
    """A longform transcriber wrapper for any shortform model.

    Longform transcriber means one being able to transcribe long audios.
    The concrete threshold between "long" and "short" audio may be
    specific for the provided :code:`shortform_model`.

    The current wrapper uses a provided segmenter to segment into
    chunks, then applies a shortform model to each chunk independently.
    If a shortform model is a
    :class:`~asr_eval.models.base.interfaces.TimedTranscriber`,
    concatenates the resulting lists for all chunks, while correcting
    the timestamps to be relative to the whole audio.

    Example:
        >>> # requires `pip install pyannote.audio>=4` for `PyannoteSegmenter`
        >>> from asr_eval.models.base.longform import LongformVAD
        >>> from asr_eval.models.pyannote_vad import PyannoteSegmenter
        >>> from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper
        >>> LongformVAD(  #doctest: +SKIP
        ...     Wav2vec2Wrapper('facebook/wav2vec2-base-960h'),
        ...     PyannoteSegmenter()
        ... )

    See also: :class:`~asr_eval.models.base.longform.LongformCTC`.
    """
    ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...