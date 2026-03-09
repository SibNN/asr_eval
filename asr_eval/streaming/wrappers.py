from collections import defaultdict
from typing import Literal, override

import numpy as np

from asr_eval.streaming.buffer import ID_TYPE
from asr_eval.streaming.model import (
    OutputChunk, Signal, StreamingASR, TranscriptionChunk
)
from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.sender import StreamingSender, get_uniform_cutoffs
from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS
from asr_eval.utils.misc import new_uid
from asr_eval.utils.audio_ops import resample


__all__ = [
    'StreamingToOffline',
    'OfflineToStreaming',
]


class StreamingToOffline(Transcriber):
    """A wrapper that turns
    :class:`~asr_eval.streaming.model.StreamingASR` into a
    :class:`~asr_eval.models.base.interfaces.Transcriber`. Transcribes
    the full audio and returns the full transcription.
    
    The :code:`StreamingASR` keeps running after :code:`.transcribe`
    and waits for new input streams. You may want to stop it via
    :code:`.streaming_model.stop_thread()` at the end.
    """
    
    def __init__(
        self, streaming_model: StreamingASR, start_thread: bool = True
    ):
        self.streaming_model = streaming_model
        if start_thread:
            self.streaming_model.start_thread()
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # as stated in Transcriber, this method accepts 16kHz waveform
        uid = new_uid()
        sampling_rate = self.streaming_model.sampling_rate
        waveform = resample(
            waveform,
            from_sampling_rate=16_000,
            to_sampling_rate=sampling_rate,
        )
        cutoffs = get_uniform_cutoffs(
            audio_length_sec=len(waveform) / sampling_rate,
            sampling_rate=sampling_rate,
        )
        sender = StreamingSender(
            id=uid,
            cutoffs=cutoffs,
            waveform=waveform,
            asr=self.streaming_model,
        )
        sender.start_sending(
            without_delays=True,
        )
        output_chunks = list(receive_transcription(
            asr=self.streaming_model, id=uid
        ))
        return TranscriptionChunk.join(output_chunks)


class OfflineToStreaming(StreamingASR):
    """Converts non-streaming (offline) ASR model into a streaming one.
    Calls the offline model with the given :code:`interval` (at the
    audio timescale).
    
    For example, let the audio be 3 seconds long, and
    :code:`interval=1`. Will call the offline model:
    
    1. On the waveform slice from 0 to 1 second (when enough data
       received)
    2. On the waveform slice from 0 to 2 seconds (when enough data
       received)
    3. On the waveform slice from 0 to 3 seconds (when enough data
       received)
    
    Each time completely overwrites the old transcription with the new
    one (this is achieved by sending a new
    :class:`~asr_eval.streaming.model.TranscriptionChunk` with the same
    id).
    
    TODO support longform audios somehow (with or without VAD).

    TODO support batching?

    TODO set also a real-time minimal interval between model calls.

    TODO add :code:`keep=True` arg to :code:`.get_with_rechunking()`
    instead making another buffer.
    """

    def __init__(self, offline_model: Transcriber, interval: float = 0.5):
        # Transcriber is hardcoded to use 16000 Hz
        super().__init__(sampling_rate=16000)
        self.offline_model = offline_model
        self.chunk_size = int(16_000 * interval)
        self.accumulated_audios: dict[ID_TYPE, FLOATS] = (
            defaultdict(lambda: np.zeros(0))
        )
    
    @override
    def _run(self):
        while True:
            id, data, is_finished, end_time = (
                self.input_buffer.get_with_rechunking(self.chunk_size)
            )
            if data is not None:
                waveform = self.accumulated_audios[id] = np.concatenate(
                    [self.accumulated_audios[id], data]
                )
                assert len(waveform) < 16_000 * 30, (
                    'Audio is too long, OfflineToStreaming'
                    ' does not support >30 sec yet'
                )
                text = self.offline_model.transcribe(waveform)
                # send always with uid=0, so that new text overwrites previous
                self.output_buffer.put(OutputChunk(
                    data=TranscriptionChunk(uid=0, text=text),
                    seconds_processed=end_time,
                ), id=id)
            if is_finished:
                self.output_buffer.put(OutputChunk(
                    data=Signal.FINISH, seconds_processed=end_time
                ), id=id)
                self.accumulated_audios.pop(id, None)
    
    @property
    @override
    def audio_type(self) -> Literal['float']:
        return 'float'