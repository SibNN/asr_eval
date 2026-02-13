from __future__ import annotations
from typing import TYPE_CHECKING, cast, override
import warnings
from io import BytesIO
import os

if TYPE_CHECKING:
    import torch
    from pyannote.audio.pipelines import VoiceActivityDetection

from asr_eval.utils.audio_ops import waveform_to_bytes
from asr_eval.models.base.interfaces import Segmenter
from asr_eval.segments.segment import AudioSegment
from asr_eval.utils.types import FLOATS


__all__ = [
    'PyannoteSegmenter',
]


class PyannoteSegmenter(Segmenter):
    '''VAD-based audio segmenter based on Pyannote. With default params
    is equivalent to :code:`gigaam.vad_utils.segment_audio`.
    
    Requires :code:`pyannote>=4.0.0`. Based on
    https://github.com/salute-developers/GigaAM/blob/main/gigaam/vad_utils.py .
    This segmenter does NOT require gigaam package to be installed,
    because all the required functions are copied from the gigaam
    package. The model is cached in PYANNOTE_CACHE dir, by default:
    ~/.cache/torch/pyannote.
    '''
    def __init__(
        self,
        min_duration: float = 15,
        max_duration: float = 22,
        strict_limit_duration: float = 30.0,
        new_chunk_threshold: float = 0.2,
        lower_limit_duration: float = 0.1,
    ):
        import torch
        
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.strict_limit_duration = strict_limit_duration
        self.new_chunk_threshold = new_chunk_threshold
        self.lower_limit_duration = lower_limit_duration
        
        self.pipeline = get_pyannote_pipeline(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    @override
    def __call__(self, waveform: FLOATS) -> list[AudioSegment]:
        segments = segment_audio(
            pipeline=self.pipeline,
            waveform=waveform,
            max_duration=self.min_duration,
            min_duration=self.max_duration,
            strict_limit_duration=self.strict_limit_duration,
            new_chunk_threshold=self.new_chunk_threshold,
        )
        return [
            seg for seg in segments
            if seg.duration >= self.lower_limit_duration
        ]


def get_pyannote_pipeline(device: torch.device | str) -> VoiceActivityDetection:
    """
    Code adopted from the GigaAM package: gigaam/vad_utils.py
    
    Retrieves a PyAnnote voice activity detection pipeline and move it
    to the specified device. The pipeline is loaded only once and reused
    across subsequent calls. It requires the Hugging Face API token to
    be set in the HF_TOKEN environment variable.
    """
    
    import torch
    from pyannote.audio import Model
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.pipelines import VoiceActivityDetection
    from torch.torch_version import TorchVersion
    
    try:
        hf_token = os.environ["HF_TOKEN"]
    except KeyError as exc:
        raise ValueError("HF_TOKEN environment variable is not set") from exc

    with torch.serialization.safe_globals(
        [
            TorchVersion,
            Problem,
            Specifications,
            Resolution,
        ]
    ):
        model = Model.from_pretrained( # type: ignore
            "pyannote/segmentation-3.0", token=hf_token
        )
    
    pipeline = VoiceActivityDetection(segmentation=model) # type: ignore
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0}) # type: ignore

    return pipeline.to(torch.device(device)) # type: ignore


def segment_audio(
    pipeline: VoiceActivityDetection,
    waveform: FLOATS,  # 16 kHz floats from -1 to 1 roughly
    sr: int = 16_000,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
) -> list[AudioSegment]:
    """
    Code adopted from the GigaAM package: gigaam/vad_utils.py, logic not
    changed.
    
    Segments an audio waveform into smaller chunks based on speech
    activity. The segmentation is performed using a PyAnnote voice
    activity detection pipeline.
    """
    
    from pyannote.core import Annotation
    from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ReproducibilityWarning)
        sad_segments = cast(Annotation, pipeline({ # type: ignore
            "uri": "filename", "audio": BytesIO(waveform_to_bytes(waveform))
        }))

    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: list[tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        boundaries.append((curr_start, curr_end))

    # Concat segments from pipeline into chunks for asr according to max/min duration
    # Segments longer than strict_limit_duration are split manually
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(waveform) / sr, segment.end)
        if curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    segments = [AudioSegment(start, end) for start, end in boundaries]
    assert all(seg.duration <= strict_limit_duration for seg in segments)
    return segments
