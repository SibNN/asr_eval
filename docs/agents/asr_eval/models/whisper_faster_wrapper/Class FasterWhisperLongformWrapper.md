# Class FasterWhisperLongformWrapper (defined in asr_eval/models/whisper_faster_wrapper.py at lines 18-256)

class FasterWhisperLongformWrapper(asr_eval.models.base.interfaces.TimedTranscriber):
    """Faster-whisper wrapper for longform transcription.

    Args:
        checkpoint: A checkpoint in CTranslate2 format. See the full
            list of available checkpoints in
            :code:`faster_whisper.transcribe.WhisperModel` docstring.
            Examples: "medium", "large-v3", "distil-large-v3",
            "large-v3-turbo". It also can be a path to a local
            checkpoint. To use custom checkpoint it needs to be
            converted to CTranslate2 format like, example:
            https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2#conversion-details
        segmenter: a segmentation method for longform transcription.
            If "internal" - faster-whisper will use
            https://github.com/snakers4/silero-vad model internally if
            :code:`segments=None` in :code:`transcribe_internal()`,
            otherwise will use the passed segments.
            If :class:`~asr_eval.models.base.interfaces.Segmenter` -
            will use the specified segmenter if :code:`segments=None` in
            :code:`transcribe_internal()`, otherwise will use the passed
            segments.
            If "shortform" - if :code:`segments=None` in
            :code:`transcribe_internal()`, will use the whole audio as a
            single segment, otherwise will use the passed segments.
            All the segments should be shorter than 30 sec.
        custom_segmenter_min_sec: Is used if `segmenter` is instance of
            :class:`~asr_eval.models.base.interfaces.Segmenter`. If the
            audio is shorter than the specified value, will not call the
            segmenter and will use the whole audio as a single segment.
        custom_segmenter_allow_merging`: If True, faster-whisper may
            internally merge several segments into one. If custom
            :code:`segments` are passed into
            :code:`transcribe_internal()`, or obtained by a custom
            segmenter passed as the :code:`segmenter` argument, then the
            length of the returned list may be larger than
            :code:`len(segments)`. Setting to False disables this
            behaviour.

    Example - 7 input segments get merged into 5 output segments
    (faster-whisper behaviour by default):

    .. code-block:: python

        waveform: FLOATS = librosa.load('tests/testdata/long.mp3', sr=16_000)[0] # type: ignore
        segments = [AudioSegment(0, 16), AudioSegment(18, 34), AudioSegment(35, 52), AudioSegment(73, 90),
                AudioSegment(91, 103), AudioSegment(103, 119), AudioSegment(120, 132)]
        model = FasterWhisperLongformWrapper(segmenter='shortform')
        outputs = model.transcribe_internal(waveform, segments=segments)
        print([(round(seg.start), round(seg.end)) for seg in outputs])

        Output: [(0, 16), (18, 34), (35, 52), (73, 103), (103, 132)]

    Example - disable merging:

    .. code-block:: python

        model = FasterWhisperLongformWrapper(segmenter='shortform', allow_merging_segments=False)
        outputs = model.transcribe_internal(waveform, segments=segments)
        print([(round(seg.start), round(seg.end)) for seg in outputs])

        Output: [(0, 8), (8, 16), (18, 34), (35, 52), (73, 90), (91, 103), (103, 119), (120, 132)]

    Example - output segments may be shorter than the corresponding
    input segments:

    .. code-block:: python

        dataset = get_dataset('multivariant-v1-200')
        segmenter = PyannoteSegmenter()
        model = FasterWhisperLongformWrapper(segmenter='shortform', allow_merging_segments=False)
        waveform = cast(FLOATS, dataset[1]['audio']['array'])
        segments = segmenter(waveform)
        print([(round(seg.start_time), round(seg.end_time)) for seg in segments])
        outputs = model.transcribe_internal(waveform, segments=segments)
        print([(round(seg.start), round(seg.end)) for seg in outputs])

        Output: [(1, 23), (23, 33), (35, 57), (57, 69), (76, 77), (78, 100), (100, 122), (122, 126), (128, 133)]
        Output: [(1, 23), (23, 32), (35, 57), (66, 69), (76, 77), (90, 91), (100, 122), (122, 126), (128, 133)]

    NOTE: For some reason, it subtrasts 0.5 sec from the original
    segments.

    NOTE: If :code:`batch_size=1` in :code:`transcribe_internal()`, and
    :code:`segmenter != 'internal'`, will call
    :code:`faster_whisper.WhisperModel` (instead of
    :code:`faster_whisper.BatchedInferencePipeline`) for each input
    segment, and then will postproces the outputs to shift all the
    output timestamps by input segment's start time.

    NOTE: If it says "Unable to load any of {libcudnn_ops.so.9.1.0,
    ...}" - then run

    .. code-block:: bash

        pip install -U nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
        sudo find / -name "libcudnn_ops.so*" 2>/dev/null

    And add the directory containing this file to LD_LIBRARY_PATH, for
    example:

    .. code-block:: bash

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib
    """
    ...

    def transcribe_internal(
        self,
        waveform: asr_eval.utils.types.FLOATS,
        segments: list[asr_eval.segments.segment.AudioSegment] | None = None,
        batch_size: int = 1,
    ) -> list[faster_whisper.transcribe.Segment]:
    ...

    @typing.override
    def timed_transcribe(
        self,
        waveform: asr_eval.utils.types.FLOATS,
        segments: list[asr_eval.segments.segment.AudioSegment] | None = None,
    ) -> list[asr_eval.segments.segment.TimedText]:
    ...