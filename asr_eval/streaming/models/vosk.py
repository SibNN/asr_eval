from collections import defaultdict
import json
from typing import override

from vosk import Model, KaldiRecognizer # type: ignore

from ..buffer import ID_TYPE
from ..model import OutputChunk, StreamingBlackBoxASR, Signal, LATEST, TranscriptionChunk

class VoskStreaming(StreamingBlackBoxASR):
    def __init__(self, model_name: str = 'vosk-model-small-en-us-0.15', sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._model = Model(model_name=model_name)
        self._recognizers: dict[ID_TYPE, KaldiRecognizer] = defaultdict(self._make_kaldi_recognizer)
        self._starting_new_part: dict[ID_TYPE, bool] = defaultdict(bool)
    
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
                self._starting_new_part.pop(id, None)
            else:
                assert isinstance(chunk.data, bytes)
                rec = self._recognizers[id]
                is_final = rec.AcceptWaveform(chunk.data) # type: ignore
                
                if is_final: # type: ignore
                    text = json.loads(rec.Result())['text'] # type: ignore
                    if self._starting_new_part[id]:
                        data = TranscriptionChunk(text=text, final=True)
                    else:
                        data = TranscriptionChunk(id=LATEST, text=text, final=True)
                    self._starting_new_part[id] = True
                else:
                    text = json.loads(rec.PartialResult())['partial'] # type: ignore
                    if self._starting_new_part[id]:
                        data = TranscriptionChunk(text=text)
                    else:
                        data = TranscriptionChunk(id=LATEST, text=text)
                    self._starting_new_part[id] = False
                self.output_buffer.put(OutputChunk(
                    data=data, n_input_chunks_processed=chunk.index + 1
                ), id=id)
                
                