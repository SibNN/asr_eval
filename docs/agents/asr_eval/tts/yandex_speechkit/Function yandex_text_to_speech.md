# Function yandex_text_to_speech (defined in asr_eval/tts/yandex_speechkit.py at lines 33-84)

def yandex_text_to_speech(
    text: str,
    api_key: str,
    voice: str | typing.Literal['random'] = 'random',
    role: str | typing.Literal['random'] = 'random',
    speed: float = 1,
    language: typing.Literal['russian', 'english'] = 'russian',
) -> tuple[asr_eval.utils.types.FLOATS, str, str]:
    """ A wrapper for speech synthesis with Yandex API v3. Will also
    work for long texts, by joining synthesized parts with pauses.

    Returns:
        Audio, voice and role.

    Raises:
        May raise grpc._channel._Rendezvous exception as said in docs.

    Installation: :code:`pip install yandex-speechkit`.

    To obtain API key, create service account and API key, as described:
    https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v2
    """
    ...