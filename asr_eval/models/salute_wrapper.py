from typing import override
import asyncio
import io
import concurrent.futures

from asr_eval.segments.segment import TimedText
from asr_eval.models.base.interfaces import TimedTranscriber
from asr_eval.utils.types import FLOATS


__all__ = [
    'SaluteWrapper',
]


class SaluteWrapper(TimedTranscriber):
    """ A wrapper for SaluteSpeech API transcriber.
    
    Need to pass api_key:
    https://developers.sber.ru/docs/ru/salutespeech/quick-start/integration-individuals
    
    Raises:
        salute_speech.exceptions.SberSpeechError: on API errors
    
    Installation: see :doc:`/guide_installation` page.
    """
    def __init__(
        self,
        api_key: str,
        format: str = 'flac',
        language: str = "en-US",  # "ru-RU" for Russian
    ):
        from salute_speech.speech_recognition import SaluteSpeechClient
         
        self.client = SaluteSpeechClient(client_credentials=api_key)
        self.format = format
        self.language = language

    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        import soundfile as sf
        
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=16000, format=self.format) # type: ignore
        buffer.seek(0)
        coroutine = self.client.audio.transcriptions.create( # type: ignore
            file=buffer,
            language=self.language,
            # config=SpeechRecognitionConfig(),
        )
        
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(lambda: asyncio.run(coroutine)).result()
        except RuntimeError:
            result = asyncio.run(coroutine)
        
        if result.segments is not None:
            segments = [
                TimedText(x.start, x.end, x.text) for x in result.segments
            ]
        elif len(result.text) > 0:
            segments = [TimedText(0, result.duration, result.text)]
        else:
            segments = []
        return segments