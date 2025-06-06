from collections import defaultdict
import json
from typing import override

from vosk import Model, KaldiRecognizer # type: ignore

from ..buffer import ID_TYPE
from ..model import OutputChunk, StreamingBlackBoxASR, Signal, LATEST, PartialTranscription

class VoskStreaming(StreamingBlackBoxASR):
    def __init__(self, model_name: str = 'vosk-model-small-en-us-0.15', sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._model = Model(model_name=model_name)
        self._recognizers: dict[ID_TYPE, KaldiRecognizer] = defaultdict(self._make_kaldi_recognizer)
    
    def _make_kaldi_recognizer(self) -> KaldiRecognizer:
        """Seems like we need to create one for each recording"""
        return KaldiRecognizer(self._model, self._sampling_rate)
    
    @override
    def _run(self):
        while True:
            chunk, id = self.input_buffer.get()
            if chunk.data is Signal.FINISH:
                self.output_buffer.put(OutputChunk(data=Signal.FINISH), id=id)
                self._recognizers.pop(id, None)
            else:
                assert isinstance(chunk.data, bytes)
                rec = self._recognizers[id]
                if rec.AcceptWaveform(chunk.data): # type: ignore
                    text = json.loads(rec.Result())['text'] # type: ignore
                    self.output_buffer.put(OutputChunk(
                        data=PartialTranscription(text=text),
                        n_input_chunks_processed=chunk.index + 1,
                    ), id=id)
                else:
                    partial_text = json.loads(rec.PartialResult())['partial'] # type: ignore
                    self.output_buffer.put(OutputChunk(
                        data=PartialTranscription(id=LATEST, text=partial_text),
                        n_input_chunks_processed=chunk.index + 1,
                    ), id=id)
                
                