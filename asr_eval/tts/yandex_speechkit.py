import io
from typing import Literal
import grpc

import librosa
import numpy as np
import yandex.cloud.ai.tts.v3.tts_pb2 as tts_pb2
import yandex.cloud.ai.tts.v3.tts_service_pb2_grpc as tts_service_pb2_grpc

from asr_eval.linguistics.linguistics import split_text_into_sentences
from asr_eval.utils.audio_ops import merge_synthetic_speech

from ..utils.types import FLOATS


__all__ = [
    'yandex_text_to_speech',
]


VOICES = {
    'alena': ['good'],
    'ermil': ['neutral'],
    'jane': ['neutral', 'good', 'evil'],
    'omazh': ['neutral', 'evil'],
    'zahar': ['neutral', 'good'],
    'jane': ['neutral', 'good', 'friendly'],
    'julia': ['neutral', 'strict'],
    'lera': ['neutral', 'friendly'],
    'masha': ['good', 'strict', 'friendly'],
    'marina': ['neutral', 'whisper', 'friendly'],
    'alexander': ['neutral', 'good'],
    'kirill': ['neutral', 'strict', 'good'],
    'anton': ['neutral', 'good'],
}

def yandex_text_to_speech(
    text: str,
    api_key: str,
    voice: str | Literal['random'] = 'random',
    role: str | Literal['random'] = 'random',
    speed: float = 1,
    language: Literal['russian', 'english'] = 'russian',
) -> tuple[FLOATS, str, str]:
    '''
    A wrapper for speech synthesis with Yandex API v3. Will also work for long texts,
    by joining synthesized parts with pauses.
    
    Installation: pip install yandex-speechkit.
    To obtain API key, create service account and API key, as described:
    https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v2
    
    Returns audio, voice and role.
    
    May raise grpc._channel._Rendezvous exception as said in docs.
    '''
    if voice == 'random':
        voice = np.random.choice(list(VOICES))
    if role == 'random':
        role = np.random.choice(VOICES[voice])
    
    MAX_SYMBOLS = 230  # API limit is 250
    
    parts = split_text_into_sentences(
        text,
        language=language,
        max_symbols=MAX_SYMBOLS,
        merge_smaller_than=MAX_SYMBOLS
    )
    
    waveforms = [
        _yandex_text_to_speech(
            text=part,
            api_key=api_key,
            voice=voice,
            role=role,
            speed=speed,
        )
        for part in parts
    ]
    
    # will do noting if there is a single part
    waveform = merge_synthetic_speech(waveforms, pause_range=(0.2, 1.2))
    
    return waveform, voice, role
    

def _yandex_text_to_speech(
    text: str,
    api_key: str,
    voice: str,
    role: str,
    speed: float = 1,
) -> FLOATS:
    request = tts_pb2.UtteranceSynthesisRequest(
        text=text, # type: ignore
        output_audio_spec=tts_pb2.AudioFormatOptions( # type: ignore
            container_audio=tts_pb2.ContainerAudio( # type: ignore
                container_audio_type=tts_pb2.ContainerAudio.WAV # type: ignore
            )
        ),
        hints=[ # type: ignore
          tts_pb2.Hints(voice=voice), # type: ignore
          tts_pb2.Hints(role=role), # type: ignore
          tts_pb2.Hints(speed=speed), # type: ignore
        ],
        loudness_normalization_type=tts_pb2.UtteranceSynthesisRequest.LUFS, # type: ignore
    )

    cred = grpc.ssl_channel_credentials() # type: ignore
    channel = grpc.secure_channel('tts.api.cloud.yandex.net:443', cred) # type: ignore
    stub = tts_service_pb2_grpc.SynthesizerStub(channel)

    it = stub.UtteranceSynthesis(request, metadata=( # type: ignore
        ('authorization', f'Api-Key {api_key}'),
    ))

    audio = io.BytesIO()
    for response in it: # type: ignore
        audio.write(response.audio_chunk.data) # type: ignore
    audio.seek(0)
    
    return librosa.load(audio, sr=16_000)[0] # type: ignore