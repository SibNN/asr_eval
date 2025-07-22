from typing import Literal, override, cast

import numpy as np
from tone import StreamingCTCPipeline

from ..streaming.buffer import ID_TYPE
from ..streaming.model import OutputChunk, StreamingASR, Signal, TranscriptionChunk


SAMPLING_RATE = 8000
CHUNK_SIZE = StreamingCTCPipeline.CHUNK_SIZE  # A 300 ms slice of audio (2400 samples)


class TOneStreaming(StreamingASR):
    def __init__(self):
        super().__init__(sampling_rate=SAMPLING_RATE)
        self.pipeline = StreamingCTCPipeline.from_hugging_face()
        self.states: dict[ID_TYPE, StreamingCTCPipeline.StateType] = {}
    
    @override
    def _run(self):
        while True:
            id, data, is_finished, end_time = self.input_buffer.get_with_rechunking(CHUNK_SIZE)
            state = self.states.get(id, None)
            if data is None: # TODO can this happen ever?
                data = np.zeros(CHUNK_SIZE, dtype=np.int32)  # as in StreamingCTCPipeline.finalize
            data = cast(StreamingCTCPipeline.InputType, data).astype(np.int32)
            if (pad_size := CHUNK_SIZE - len(data)) > 0:
                # TODO add option for padding in get_with_rechunking
                data = np.concatenate([data, np.zeros(pad_size, dtype=np.int32)])
            output, state = self.pipeline.forward(data, state, is_last=is_finished)
            self.states[id] = state
            for utterance in output:
                # start, end time are currently not used
                self.output_buffer.put(OutputChunk(
                    data=TranscriptionChunk(text=utterance.text),
                    seconds_processed=end_time,
                ), id=id)
            if is_finished:
                self.states.pop(id, None)
                self.output_buffer.put(OutputChunk(data=Signal.FINISH, seconds_processed=end_time), id=id)
                
    @property
    @override
    def audio_type(self) -> Literal['float', 'int', 'bytes']:
        return 'int'