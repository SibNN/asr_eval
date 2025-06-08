from collections import defaultdict
import json
from typing import override

from vosk import Model, KaldiRecognizer # type: ignore

from ..buffer import ID_TYPE
from ..model import AUDIO_CHUNK_TYPE, OutputChunk, StreamingBlackBoxASR, Signal, LATEST, TranscriptionChunk

class VoskStreaming(StreamingBlackBoxASR):
    def __init__(
        self,
        model_name: str = 'vosk-model-small-en-us-0.15',
        sampling_rate: int = 16_000,
        chunk_length_sec: float | None = None,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self._model = Model(model_name=model_name)
        self._recognizers: dict[ID_TYPE, KaldiRecognizer] = defaultdict(self._make_kaldi_recognizer)
        self._next_chunk_starts_new_part: dict[ID_TYPE, bool] = defaultdict(bool)
        self.chunk_size = (
            int(sampling_rate * 2 * chunk_length_sec)  #  * 2 because we receive 2 bytes each frame
            if chunk_length_sec is not None
            else None
        )
    
    def _make_kaldi_recognizer(self) -> KaldiRecognizer:
        """Seems like we need to create one for each recording"""
        return KaldiRecognizer(self._model, self._sampling_rate)
    
    def _send_finish(self, id: ID_TYPE):
        self.output_buffer.put(OutputChunk(data=Signal.FINISH), id=id)
        self._recognizers.pop(id, None)
        self._next_chunk_starts_new_part.pop(id, None)
    
    def _process_chunk(self, id: ID_TYPE, data: AUDIO_CHUNK_TYPE, end_time: float | None):
        assert isinstance(data, bytes)
        rec = self._recognizers[id]
        is_final = rec.AcceptWaveform(data) # type: ignore
        text = (
            json.loads(rec.Result())['text'] # type: ignore
            if is_final
            else json.loads(rec.PartialResult())['partial'] # type: ignore
        )
        output_data = (
            TranscriptionChunk(text=text)
            if self._next_chunk_starts_new_part[id]
            else TranscriptionChunk(ref=LATEST, text=text)
        )
        self._next_chunk_starts_new_part[id] = is_final
        self.output_buffer.put(OutputChunk(
            data=output_data,
            seconds_processed=end_time,
        ), id=id)
    
    @override
    def _run(self):
        if self.chunk_size is None:
            # run without rechunking
            while True:
                chunk, id = self.input_buffer.get()
                if chunk.data is Signal.FINISH:
                    self._send_finish(id)
                else:
                    self._process_chunk(id, chunk.data, chunk.end_time)
        else:
            # run with rechunking
            while True:
                id, data, is_finished, end_time = self.input_buffer.get_with_rechunking(self.chunk_size)
                if data is not None:
                    self._process_chunk(id, data, end_time)
                if is_finished:
                    self._send_finish(id)