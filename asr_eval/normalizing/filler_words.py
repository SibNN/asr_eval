import re

__all__ = [
    'RuFillerRemover',
]


class RuFillerRemover:
    """Removes Russian filler words."""
    
    def __init__(self):
        self.to_remove = {
            'вот',
            'ну',
            'э',
            'ээ',
            'эээ',
            'аа',
            'ааа',
            'ха',
            'хаха',
            'хахаха',
            'ага',
            'угу',
            'ой',
            'ай',
            'ух',
            'ах',
            'фу',
            'вау',
            'бля',
            'нах',
        }
    def __call__(self, text: str) -> str:
        for match in list(re.finditer(r'\w+', text))[::-1]:
            word = match.group().lower()
            if word in self.to_remove:
                text = text[:match.start()] + ' ' + text[match.end():]
        return text