import re

from asr_eval.linguistics.linguistics import get_all_declensions


__all__ = [
    'TranslitNormalizer',
]


RU_WORD_MAPPING = {
    'facebook': 'фейсбук',
    'фэйсбук': 'фейсбук',
    'iphone': 'айфон',
    'android': 'андроид',
    'samsung': 'самсунг',
    'youtube': 'ютуб',
    'ютьюб': 'ютуб',
    'ютюб': 'ютуб',
    'telegram': 'телеграм',
    'телеграмм': 'телеграм',
    'google': 'гугл',
    'twitter': 'твиттер',
    'skill': 'скилл',
    'скил': 'скилл',
    'kamaz': 'камаз',
    'binance': 'бинанс',
    'security': 'секьюрити',
    'yandex': 'яндекс',
    'sber': 'сбер',
    'ios': 'айос',
    'mail': 'мэйл',
    'мейл': 'мэйл',
    'ru': 'ру',
    'instagram': 'инстаграм',
    'инстаграмм': 'инстаграм',
    'minecraft': 'майнкрафт',
    'bmw': 'бмв',
    'mercedes': 'мерседес',
    'toyota': 'тойота',
    'ok': 'ок',
    'окей': 'ок',
    'блять': 'блядь',
}


class TranslitNormalizer:
    """
    Normalizes several Russian transliterated words using a pre-defined
    rules.
    
    Example:
        >>> print(TranslitNormalizer()('В Facebook и твиттере'))
        В фейсбук и твиттер
    """
    
    def __init__(
        self,
        word_mapping: dict[str, str] = RU_WORD_MAPPING,
    ):
        from pymorphy3 import MorphAnalyzer

        self._morph = MorphAnalyzer(path=None)

        self.word_mapping = word_mapping

        additional_replacements: dict[str, str] = {}
        for word_from, word_to in self.word_mapping.items():
            for infl in get_all_declensions(word_to):
                additional_replacements[infl] = word_to
            for infl in get_all_declensions(word_from):
                additional_replacements[infl] = word_to

        self.word_mapping |= additional_replacements
        
    def __call__(self, text: str) -> str:
        for match in list(re.finditer(r'\w+', text))[::-1]:
            word = match.group().lower()
            if (replacement := self.word_mapping.get(word, None)) is not None:
                text = text[:match.start()] + replacement + text[match.end():]
        return text