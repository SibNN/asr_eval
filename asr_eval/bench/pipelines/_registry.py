from __future__ import annotations
from functools import cache
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from asr_eval.models.base.interfaces import TimedTranscriber, Transcriber
from asr_eval.bench.datasets import AudioSample
from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import remap_time, make_sender
from asr_eval.streaming.model import Signal, StreamingASR, TranscriptionChunk
from asr_eval.utils.timer import Timer


pipelines_registry: dict[str, type[TranscriberPipeline]] = {}
"""A mapping from the assigned unique name to the pipeline class."""


class TranscriberPipeline(ABC):
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
    def __init__(self, warmup: bool = False):
        self.transcriber = self.init()
        # start thread for StreamingASR
        if isinstance(self.transcriber, StreamingASR):
            self.transcriber.start_thread()
        if warmup:
            # do a warmup run to estimate samples runtime correctly (may lag
            # on the first sample)
            print('doing a warmup...', flush=True)
            waveform = np.zeros(16_000 * 5, dtype=np.float32)
            if isinstance(self.transcriber, StreamingASR):
                _cutoffs, sender = make_sender(waveform, self.transcriber)
                sender.start_sending(without_delays=True)
                list(receive_transcription(
                    asr=self.transcriber, id=sender.id
                ))
            else:
                self.transcriber.transcribe(waveform)
    
    @abstractmethod
    def init(self) -> Transcriber | StreamingASR: ...
    
    def run(self, sample: AudioSample) -> dict[str, Any]:
        waveform = sample['audio']['array']
        outputs: dict[str, Any] = {}
        with Timer() as timer:
            if isinstance(self.transcriber, StreamingASR):
                do_remapping = not self.transcriber.is_multithreaded
                cutoffs, sender = make_sender(waveform, self.transcriber)
                sender.start_sending(without_delays=do_remapping)
                output_chunks = list(receive_transcription(
                    asr=self.transcriber, id=sender.id
                ))
                input_chunks = sender.join_and_get_history()
                if do_remapping:
                    input_chunks, output_chunks = remap_time(
                        cutoffs, input_chunks, output_chunks
                    )
                for input_chunk in input_chunks:
                    if input_chunk.data is not Signal.FINISH:
                        input_chunk.data = None # type: ignore
                outputs['cutoffs'] = cutoffs
                outputs['input_chunks'] = input_chunks
                outputs['output_chunks'] = output_chunks
                outputs['text'] = TranscriptionChunk.join(output_chunks)
            elif isinstance(self.transcriber, TimedTranscriber):
                outputs['timed_text'] = (
                    self.transcriber.timed_transcribe(waveform)
                )
                outputs['text'] = ' '.join([
                    seg.text for seg in outputs['timed_text']
                ])
            else:
                outputs['text'] = self.transcriber.transcribe(waveform)
        outputs['elapsed_time'] = timer.elapsed_time
        return outputs
    
    def __init_subclass__(cls, register_as: str | None = None, **kwargs: Any):
        global pipelines_registry
        super().__init_subclass__(**kwargs)
        if register_as:
            assert register_as not in pipelines_registry
            pipelines_registry[register_as] = cls
            cls.name = register_as


def get_pipeline(name: str) -> type[TranscriberPipeline]:
    """Get a registered pipeline class."""
    if name not in pipelines_registry:
        raise ValueError(f'Pipeline does not exist: {name}')
    return pipelines_registry[name]


@cache
def get_pipeline_index(name: str) -> int:
    """Get an index (in registration order) for a registered pipeline,
    or -1 if not registered.
    """
    return (
        list(pipelines_registry).index(name)
        if name in pipelines_registry
        else -1
    )