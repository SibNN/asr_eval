from collections import defaultdict
from typing import Literal, override

import numpy as np

from .buffer import ID_TYPE
from .model import OutputChunk, Signal, StreamingASR, TranscriptionChunk, prepare_audio_format
from .caller import receive_full_transcription
from .sender import StreamingAudioSender
from ..models.base.interfaces import Transcriber
from ..utils.types import FLOATS
from ..utils.misc import new_uid


class Offline(Transcriber):
    '''
    A wrapper that turns StreamingASR into a Transcriber. Transcribes
    the full audio and returns the full transcription.
    '''
    def __init__(self, streaming_model: StreamingASR):
        self.streaming_model = streaming_model
        self.streaming_model.start_thread()
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        id = new_uid()
        audio, array_len_per_sec = prepare_audio_format(
            waveform, self.streaming_model, sampling_rate=16_000
        )
        sender = StreamingAudioSender(
            id=id, audio=audio, array_len_per_sec=array_len_per_sec, track_history=False
        )
        sender.send_all_without_delays(self.streaming_model.input_buffer)
        output_chunks = receive_full_transcription(asr=self.streaming_model, id=id)
        return TranscriptionChunk.join(output_chunks)


class QuasiStreaming(StreamingASR):
    def __init__(self, offline_model: Transcriber, interval: float = 0.5):
        # TODO add keep=True arg to .get_with_rechunking() instead making of another buffer
        # TODO support batching?
        super().__init__()
        self.offline_model = offline_model
        self.chunk_size = int(16_000 * interval)
        self.accumulated_audios: dict[ID_TYPE, FLOATS] = defaultdict(lambda: np.zeros(0))
    
    @override
    def _run(self):
        while True:
            id, data, is_finished, end_time = self.input_buffer.get_with_rechunking(self.chunk_size)
            if data is not None:
                self.accumulated_audios[id] = np.concatenate([self.accumulated_audios[id], data])
                text = self.offline_model.transcribe(self.accumulated_audios[id])
                # send always with uid=0, so that new text overwrites previous
                self.output_buffer.put(OutputChunk(
                    data=TranscriptionChunk(uid=0, text=text), seconds_processed=end_time
                ), id=id)
            if is_finished:
                self.output_buffer.put(OutputChunk(
                    data=Signal.FINISH, seconds_processed=end_time
                ), id=id)
                self.accumulated_audios.pop(id, 0)
    
    @property
    @override
    def audio_type(self) -> Literal['float', 'int', 'bytes']:
        return 'float'