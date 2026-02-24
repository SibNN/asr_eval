# Class YandexSpeechKitWrapper (defined in asr_eval/models/yandex_speechkit_wrapper.py at lines 15-119)

class YandexSpeechKitWrapper(asr_eval.models.base.interfaces.TimedTranscriber):
    """A wrapper for Yandex SpeechKit transcriber.

    Docs: https://yandex.cloud/ru/docs/speechkit/stt/models

    To obtain API key, create service account and API key, as described:
    https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v2

    Speechkit provides timings for each word, raw and normalized text,
    it seems to normalize text for language='ru-Ru' but not for
    language='auto'.

    Example raw:

    .. code-block:: none

        [седьмого [0.399, 1.060], восьмого [1.120, 1.780], мая [1.860, 2.399], в [2.520, 2.580],
        пуэрто [2.639, 3.340], рико [3.419, 3.899], прошел [4.110, 4.680], шестнадцатый [4.839, 5.839],
        этап [5.890, 6.299], формулы [6.470, 7.170], один [7.259, 7.740], с [7.859, 7.890],
        фондом [8.040, 8.780], сто [8.950, 9.320], тысяч [9.429, 9.690], долларов [9.900, 10.700],
        победителем [11.559, 12.346], стал [12.420, 12.733],

    Example normalized:

    .. code-block:: none

        7 8 Мая в Пуэрто Рико прошел 16 этап Формулы 1 с Фондом 10.00000000000% $-победителем стал

    As you can see, normalization introduces some errors, and it is
    sometimes hard to align raw and normalized text.

    If :code:`normalize=True` and normalized text is returned by the
    API:

    1. :code:`transcribe()` returns a full normalized text.
    2. :code:`timed_transcribe()` returns a list of normalized
       utterances if available, otherwise a fill text.

    Otherwise:
    1. :code:`transcribe()` returns a full unnormalized text.
    2. :code:`timed_transcribe()` returns a list of unnormalized single
       words.

    Authors: Dmitry Ezhov & Oleg Sedukhin
    """
    ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...