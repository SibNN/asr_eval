from collections import defaultdict
import json
from typing import Literal, override, cast

from vosk import Model, KaldiRecognizer # type: ignore

from ..streaming.buffer import ID_TYPE
from ..streaming.model import AUDIO_CHUNK_TYPE, OutputChunk, StreamingASR, Signal, TranscriptionChunk
from ..utils.misc import new_uid


class VoskStreaming(StreamingASR):
    def __init__(
        self,
        model_name: str = 'vosk-model-small-en-us-0.15',
        sampling_rate: int = 16_000,
        chunk_length_sec: float | None = None,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self._model = Model(model_name=model_name)
        self._recognizers: dict[ID_TYPE, KaldiRecognizer] = defaultdict(self._make_kaldi_recognizer)
        self._last_was_partial: dict[ID_TYPE, bool] = defaultdict(lambda: False)
        self._transcription_chunk_ref: dict[ID_TYPE, str] = defaultdict(new_uid)
        self.chunk_size = (
            int(sampling_rate * 2 * chunk_length_sec)  #  * 2 because we receive 2 bytes each frame
            if chunk_length_sec is not None
            else None
        )
    
    def _make_kaldi_recognizer(self) -> KaldiRecognizer:
        """Seems like we need to create one for each recording"""
        return KaldiRecognizer(self._model, self.sampling_rate)
    
    def _send_transcription_chunk(self, id: ID_TYPE, is_final: bool, seconds_processed: float):
        rec = self._recognizers[id]
        if is_final:
            # final text part
            text = json.loads(rec.Result())['text'] # type: ignore
        else:
            # non-final text part
            text = json.loads(rec.PartialResult())['partial'] # type: ignore
        
        self.output_buffer.put(OutputChunk(
            data=TranscriptionChunk(uid=self._transcription_chunk_ref[id], text=text),
            seconds_processed=seconds_processed,
        ), id=id)
        
    def _send_finish(self, id: ID_TYPE, end_time: float):
        if self._last_was_partial[id]:
            self._send_transcription_chunk(id, True, end_time)
        self.output_buffer.put(OutputChunk(data=Signal.FINISH, seconds_processed=end_time), id=id)
        self._recognizers.pop(id, None)
        self._transcription_chunk_ref.pop(id, None)
        self._last_was_partial.pop(id, None)
    
    def _process_chunk(self, id: ID_TYPE, data: AUDIO_CHUNK_TYPE, end_time: float):
        assert isinstance(data, bytes)
        rec = self._recognizers[id]
        
        is_final = cast(bool, rec.AcceptWaveform(data)) # type: ignore
        self._send_transcription_chunk(id, is_final, end_time)
        if is_final:
            self._transcription_chunk_ref[id] = new_uid()
        self._last_was_partial[id] = not is_final
    
    @override
    def _run(self):
        if self.chunk_size is None:
            # run without rechunking
            while True:
                chunk, id = self.input_buffer.get()
                if chunk.data is Signal.FINISH:
                    # text = json.loads(self._recognizers[id].Result())['text'] # type: ignore
                    # print('final finish', text)
                    self._send_finish(id, chunk.end_time)
                else:
                    self._process_chunk(id, chunk.data, chunk.end_time)
        else:
            # run with rechunking
            while True:
                id, data, is_finished, end_time = self.input_buffer.get_with_rechunking(self.chunk_size)
                if data is not None:
                    self._process_chunk(id, data, end_time)
                if is_finished:
                    self._send_finish(id, end_time)
    
    @property
    @override
    def audio_type(self) -> Literal['float', 'int', 'bytes']:
        return 'bytes'