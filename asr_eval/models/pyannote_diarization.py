from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, cast
import warnings
import yaml
import os
from pprint import pprint

if TYPE_CHECKING:
    from pyannote.core.annotation import Annotation
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

from asr_eval.segments.segment import DiarizationSegment
from asr_eval.utils.types import FLOATS


__all__ = [
    'PyannoteDiarizationWrapper',
]


class PyannoteDiarizationWrapper:
    """A wrapper for Pyannote diarization.
    
    Requires :code:`pyannote>=4.0.0`. To use, you first need to accept
    conditions here
    https://huggingface.co/pyannote/speaker-diarization-community-1 ,
    then specify your HF_TOKEN in the environmental variable.
    """
    
    pipeline: SpeakerDiarization
    model_name: str

    # Pyannote models can be instantiated in 3 ways:
    # - from config as string, like "pyannote/speaker-diarization-community-1"
    # - from config as dict
    # - by calling class constructors (see examples in this file)

    def __init__(
        self,
        verbose: bool = False,
        preset: str = 'pyannote/speaker-diarization-community-1',
        segmentation: str | None = None,  # if not set, use default from preset
        embedding: str | None = None,  # if not set, use default from preset
    ):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', FutureWarning)

            import torch
            import pyannote.audio
            from pyannote.audio.utils.reproducibility import (
                ReproducibilityWarning
            )
            self.ReproducibilityWarning = ReproducibilityWarning
            from pyannote.audio.pipelines import SpeakerDiarization # type: ignore
            from pyannote.audio.utils.hf_hub import (
                AssetFileName, download_from_hf_hub
            )
            from pyannote.audio.core.pipeline import expand_subfolders # type: ignore
            
            self.model_name = preset

            # a version from https://huggingface.co/pyannote/speaker-diarization-community-1
            # config: https://huggingface.co/pyannote/speaker-diarization-community-1/blob/main/config.yaml
            # since there are 'params' field in config, the pipeline is
            # instantiated automatically; it supports exclusive speaker
            # diarization (is used here) see
            # https://www.pyannote.ai/blog/precision-2
            # sec. "Reconciliation of speech-to-text/diarization timestamps"
            # and https://huggingface.co/pyannote/speaker-diarization-community-1#exclusive-speaker-diarization
            
            # what happens in the code below:
            
            # we want to get the config from `preset` but modify
            # `segmentation` and `embedding` fields the default
            # `pyannote.audio.Pipeline.from_pretrained` method does not allow
            # this so we first download the preset config using
            # `download_from_hf_hub`, then read end modity fields, then pass
            # the config as dict into `Pipeline.from_pretrained`;  however, if
            # config is passed as dict, the `Pipeline.from_pretrained` method
            # incorrectly expands $model/{subfolder}-like entries in config:
            # it treats them as local paths and not the HF paths, which leads
            # to "No such file or directory" to fix this, we call
            # `expand_subfolders` manually before `Pipeline.from_pretrained`

            cache_dir = Path.home() / '.cache/pyannote'

            config_yml = cast(str, download_from_hf_hub(
                preset,
                AssetFileName.Pipeline,
                cache_dir=cache_dir,
            ))

            with open(config_yml, "r") as fp:
                config = yaml.load(fp, Loader=yaml.SafeLoader)
            
            if segmentation is not None:
                config['pipeline']['params']['segmentation'] = segmentation
            
            if embedding is not None:
                config['pipeline']['params']['embedding'] = embedding
                
            expand_subfolders(
                config,
                model_id=preset,
                parent_revision=None,
                token=None,
                cache_dir=cache_dir,
            )

            if verbose:
                pprint(config, indent=4)

            cwd = Path().cwd()
            os.chdir(Path(config_yml).parent)
            self.pipeline = cast(
                SpeakerDiarization,
                pyannote.audio.Pipeline.from_pretrained( # type: ignore
                    config, cache_dir=cache_dir,
                )
            )
            os.chdir(cwd)
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device('cuda')) # type: ignore
            
    
    def __call__(
        self, mono_sound: FLOATS, sampling_rate: int = 16_000
    ) -> tuple[list[DiarizationSegment], FLOATS]:
        import torch
        
        annotation: Annotation

        with warnings.catch_warnings():
            # "TensorFloat-32 (TF32) has been disabled"
            warnings.simplefilter('ignore', self.ReproducibilityWarning)
            # "std(): degrees of freedom is <= 0"
            warnings.simplefilter('ignore', UserWarning)

            outputs: DiarizeOutput = self.pipeline.apply( # type: ignore
                {
                    'waveform': (
                        torch.tensor(mono_sound, dtype=torch.float32)
                        .unsqueeze(0)
                    ),
                    'sample_rate': sampling_rate,
                },
                num_speakers=None,
                min_speakers=None,
                max_speakers=None,
            )
        
        annotation = cast(Annotation, outputs.exclusive_speaker_diarization) # type: ignore
        segments = [
            DiarizationSegment(
                start_time=float(turn.start),
                end_time=float(turn.end),
                speaker=int(annotation.labels().index(speaker)),
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True) # type: ignore
        ]
        
        assert outputs.speaker_embeddings is not None # type: ignore
        return segments, outputs.speaker_embeddings # type: ignore


# examples of initializing pyannote diarization using class constructors:


# pyannote/speaker-diarization-community-1
#     + pyannote/wespeaker-voxceleb-resnet34-LM

# model.pipeline = SpeakerDiarization(
#     segmentation=Model.from_pretrained( # type: ignore
#         checkpoint='pyannote/speaker-diarization-community-1',
#         subfolder='segmentation',
#         cache_dir=Path('/home/oleg/.cache/pyannote'),
#     ),
#     segmentation_step=0.1,
#     embedding=Model.from_pretrained( # type: ignore
#         checkpoint='pyannote/wespeaker-voxceleb-resnet34-LM',
#         cache_dir=Path('/home/oleg/.cache/pyannote'),
#     ),
#     embedding_exclude_overlap=True,
#     plda=PLDA.from_pretrained( # type: ignore
#         checkpoint='pyannote/speaker-diarization-community-1',
#         subfolder='plda',
#         cache_dir=Path('/home/oleg/.cache/pyannote'),
#     ),
#     clustering='VBxClustering',
#     embedding_batch_size=32,
#     segmentation_batch_size=32,
#     cache_dir=Path('/home/oleg/.cache/pyannote'),
# ).to(torch.device('cuda'))
# model.pipeline.instantiate({
#     'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8},
#     'segmentation': {'min_duration_off': 0.0},
# })
# model.pipeline.instantiate({
#     'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8},
#     'segmentation': {'min_duration_off': 0.0},
# })


# pyannote/speaker-diarization-3.1 + speechbrain/spkrec-ecapa-voxceleb

# model.pipeline = SpeakerDiarization(
#     segmentation='pyannote/segmentation-3.0',
#     segmentation_step=0.1,
#     embedding='speechbrain/spkrec-ecapa-voxceleb',
#     embedding_exclude_overlap=True,
#     plda={'checkpoint': 'pyannote/speaker-diarization-community-1', 'subfolder': 'plda'}, # not used
#     clustering='AgglomerativeClustering',
#     embedding_batch_size=32,
#     segmentation_batch_size=32,
#     der_variant=None,
#     token=None,
#     cache_dir=Path('/home/oleg/.cache/pyannote'),
# ).to(torch.device('cuda'))
# model.pipeline.instantiate({
#     'clustering': {'method': 'centroid', 'min_cluster_size': 12, 'threshold': 0.7045654963945799},
#     'segmentation': {'min_duration_off': 0.0}
# })


# pyannote/speaker-diarization-3.1 + pyannote/wespeaker-voxceleb-resnet34-LM

# model.pipeline = SpeakerDiarization(
#     segmentation='pyannote/segmentation-3.0',
#     segmentation_step=0.1,
#     embedding='pyannote/wespeaker-voxceleb-resnet34-LM',
#     embedding_exclude_overlap=True,
#     plda={'checkpoint': 'pyannote/speaker-diarization-community-1', 'subfolder': 'plda'}, # not used
#     clustering='AgglomerativeClustering',
#     embedding_batch_size=32,
#     segmentation_batch_size=32,
#     der_variant=None,
#     token=None,
#     cache_dir=Path('/home/oleg/.cache/pyannote'),
# ).to(torch.device('cuda'))
# model.pipeline.instantiate({
#     'clustering': {'method': 'centroid', 'min_cluster_size': 12, 'threshold': 0.7045654963945799},
#     'segmentation': {'min_duration_off': 0.0}
# })