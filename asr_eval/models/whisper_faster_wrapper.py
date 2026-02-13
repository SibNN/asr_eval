from __future__ import annotations
from typing import TYPE_CHECKING, Literal, override
import numpy as np

if TYPE_CHECKING:
    import faster_whisper.transcribe

from asr_eval.models.base.interfaces import Segmenter, TimedTranscriber
from asr_eval.segments.segment import AudioSegment, TimedText
from asr_eval.utils.types import FLOATS


__all__ = [
    'FasterWhisperLongformWrapper',
]


class FasterWhisperLongformWrapper(TimedTranscriber):
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
    
    def __init__(
        self,
        device: Literal['cuda', 'cpu', 'auto'] = 'auto',
        checkpoint: str = 'large-v3-turbo',
        segmenter: Literal['internal', 'shortform'] | Segmenter = 'internal',
        custom_segmenter_min_sec: float = 30,
        allow_merging_segments: bool = True,
        dtype: Literal[
            'float16',
            'float32',
            'bfloat16',
            'int8_float16',
            'int8_bfloat16',
            'int8_float32',
            'int8',
        ] = 'float16',
    ):
        from faster_whisper.vad import collect_chunks # type: ignore
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        
        import faster_whisper.transcribe
        self.transcribe_module = faster_whisper.transcribe
        
        self.model = WhisperModel(
            checkpoint, device=device, compute_type=dtype
        )
        self.batched_model = BatchedInferencePipeline(model=self.model)

        self.segmenter: (
            Literal['internal', 'shortform'] | Segmenter
        ) = segmenter
        self.custom_segmenter_min_sec = custom_segmenter_min_sec
        self.allow_merging_segments = allow_merging_segments

        self.orig_collect_chunks = faster_whisper.transcribe.collect_chunks # type: ignore
    
    def transcribe_internal(
        self,
        waveform: FLOATS,
        segments: list[AudioSegment] | None = None,
        batch_size: int = 1,
    ) -> list[faster_whisper.transcribe.Segment]:
        waveform = waveform.astype(np.float32)
        
        if segments is None:
            audio_len = len(waveform) / 16_000
            match self.segmenter:
                case 'internal':
                    pass
                case Segmenter():
                    if audio_len < self.custom_segmenter_min_sec:
                        segments = [AudioSegment(0, audio_len)]
                    else:
                        segments = self.segmenter(waveform)
                case 'shortform':
                    assert audio_len < 30, (
                        'Audio length should be <= 30'
                        ' for shortform transcription'
                    )
                    segments = [AudioSegment(0, audio_len)]
                    
        # use self.batched_model or self.model?
        use_batched_inference = batch_size > 1 or self.segmenter == 'internal'
        
        if use_batched_inference:
            print('Using batched inference')
            clip_timestamps = (
                [
                    {"start": seg.start_time, "end": seg.end_time}
                    for seg in segments
                ]
                if segments is not None
                else None
            )
                
            if clip_timestamps is not None and len(clip_timestamps) > 1:
                if not self.allow_merging_segments:
                    self.transcribe_module.collect_chunks = collect_chunks_stub

            outputs_generator, _info = self.batched_model.transcribe( # type: ignore
                waveform,
                clip_timestamps=clip_timestamps,
                batch_size=batch_size,
                word_timestamps=True,
            )

            outputs = list(outputs_generator)

            self.transcribe_module.collect_chunks = self.orig_collect_chunks # type: ignore
        
        else:
            print('Using unbatched inference')
            assert segments is not None
            
            outputs: list[faster_whisper.transcribe.Segment] = []
            
            for input_segment in segments:
                outputs_generator, _info = self.model.transcribe( # type: ignore
                    waveform[input_segment.slice(16_000)],
                    word_timestamps=True,
                )
                outputs_for_input_segment = list(outputs_generator)
            
                # updating timestamps
                for output_segment in outputs_for_input_segment:
                    dt = input_segment.start_time
                    output_segment.start += dt
                    output_segment.end += dt
                    output_segment.seek += int(dt * 16_000)
                    if output_segment.words:
                        for word in output_segment.words:
                            word.start += dt
                            word.end += dt
                
                outputs += outputs_for_input_segment

        return outputs

    @override
    def timed_transcribe(
        self,
        waveform: FLOATS,
        segments: list[AudioSegment] | None = None,
    ) -> list[TimedText]:
        outputs = self.transcribe_internal(waveform, segments)
        return [
            TimedText(
                start_time=float(seg.start),
                end_time=float(seg.end),
                text=seg.text.strip()
            )
            for seg in outputs
        ]


def collect_chunks_stub( # type: ignore
    audio: np.ndarray, # type: ignore
    chunks: list[dict], # type: ignore
    sampling_rate: int = 16000,
    max_duration: float = float("inf"),
) -> tuple[list[np.ndarray], list[dict[str, float]]]: # type: ignore
    '''
    A replacement for faster_whisper.vad.collect_chunks,
    so that faster-whisper will not merge chunks.
    (`max_duration` argument is not used)
    '''
    if not chunks:
        chunk_metadata = { # type: ignore
            "offset": 0,
            "duration": 0,
            "segments": [],
        }
        return [np.array([], dtype=np.float32)], [chunk_metadata] # type: ignore

    audio_chunks = []
    chunks_metadata = []
    total_duration = 0

    for chunk in chunks: # type: ignore
        audio_chunks.append(audio[chunk["start"]:chunk["end"]]) # type: ignore
        current_duration = chunk["end"] - chunk["start"] # type: ignore
        chunk_metadata = { # type: ignore
            "offset": total_duration / sampling_rate,
            "duration": current_duration / sampling_rate,
            "segments": [chunk],
        }
        total_duration += current_duration # type: ignore
        chunks_metadata.append(chunk_metadata) # type: ignore

    return audio_chunks, chunks_metadata # type: ignore





