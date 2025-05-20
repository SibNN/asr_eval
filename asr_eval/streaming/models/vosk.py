from collections import defaultdict
import json
from typing import override

from vosk import Model, KaldiRecognizer # type: ignore

from ..transcription import PartialTranscription
from ..model import StreamingBlackBoxASR, Signal, RECORDING_ID_TYPE

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
            id, chunk = self.input_buffer.get()
            if chunk is Signal.EXIT:
                self._recognizers = {}
                return
            elif chunk is Signal.FINISH:
                self.output_buffer.put((id, Signal.FINISH))
                self._recognizers.pop(id, None)
            else:
                assert isinstance(chunk, bytes)
                rec = self._recognizers[id]
                if rec.AcceptWaveform(chunk): # type: ignore
                    text = json.loads(rec.Result())['text'] # type: ignore
                    self.output_buffer.put((id, PartialTranscription(text=text)))
                else:
                    partial_text = json.loads(rec.PartialResult())['partial'] # type: ignore
                    self.output_buffer.put((id, PartialTranscription(id='latest', text=partial_text)))
                
                