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


class StreamingToOffline(Transcriber):
    '''
    A wrapper that turns StreamingASR into a Transcriber. Transcribes
    the full audio and returns the full transcription.
    
    Don't forget to call model.streaming_model.stop_thread() at the end.
    '''
    def __init__(self, streaming_model: StreamingASR, start_thread: bool = True):
        self.streaming_model = streaming_model
        if start_thread:
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


class OfflineToStreaming(StreamingASR):
    '''
    Converts non-streaming (offline) ASR model into a streaming one. Calls the offline model
    with the given `interval` (at the audio timescale).
    
    For example, let the audio be 3 seconds long, and interval=1. Will call the offline model:
    
    1. On the waveform slice from 0 to 1 second (when enough data received)
    2. On the waveform slice from 0 to 2 seconds (when enough data received)
    3. On the waveform slice from 0 to 3 seconds (when enough data received)
    
    Each time overwrites the old transcription with the new one (this is achieved by
    sending a new TranscriptionChunk with the same uid).
    
    TODO support batching?
    TODO set also a real-time minimal interval between model calls
    TODO add keep=True arg to .get_with_rechunking() instead making of another buffer
    '''
    def __init__(self, offline_model: Transcriber, interval: float = 0.5):
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