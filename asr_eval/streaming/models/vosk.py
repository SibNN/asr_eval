from collections import defaultdict
import json
from typing import override

from vosk import Model, KaldiRecognizer # type: ignore

from ..transcription import LATEST, PartialTranscription
from ..model import OutputChunk, StreamingBlackBoxASR, Signal, RECORDING_ID_TYPE

class VoskStreaming(StreamingBlackBoxASR):
    def __init__(self, sampling_rate: int = 16_000):
        super().__init__(sampling_rate=sampling_rate)
        self._model = Model(lang='en-us')
        self._recognizers: dict[RECORDING_ID_TYPE, KaldiRecognizer] = defaultdict(self._make_kaldi_recognizer)
    
    def _make_kaldi_recognizer(self) -> KaldiRecognizer:
        """Seems like we need to create one for each recording"""
        return KaldiRecognizer(self._model, self._sampling_rate)
    
    @override
    def _run(self):
        while True:
            input_chunk = self.input_buffer.get()
            id = input_chunk.id
            
            if input_chunk.data is Signal.EXIT:
                self._recognizers = {}
                return
            elif input_chunk.data is Signal.FINISH:
                self.output_buffer.put(OutputChunk(id=id, data=Signal.FINISH))
                self._recognizers.pop(id, None)
            else:
                assert isinstance(input_chunk.data, bytes)
                rec = self._recognizers[id]
                if rec.AcceptWaveform(input_chunk.data): # type: ignore
                    text = json.loads(rec.Result())['text'] # type: ignore
                    self.output_buffer.put(OutputChunk(id=id, data=PartialTranscription(text=text)))
                else:
                    partial_text = json.loads(rec.PartialResult())['partial'] # type: ignore
                    self.output_buffer.put(OutputChunk(id=id, data=PartialTranscription(id=LATEST, text=partial_text)))
                
                