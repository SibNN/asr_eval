from typing import cast, override
import string

from asr_eval.align.solvers.dynprog import solve_optimal_alignment
from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.transcription import Token
from asr_eval.linguistics.linguistics import word_freq
from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS


class RuWordFreqComparator(Transcriber):
    """A composite transcriber that retrieves prediction for two models
    (where the first is generally better) and replaces words predicted
    by the first model by words predicted by the second model in cases
    where the second model's word is more frequent in language.
    
    Works for Russian language currently.
    """
    def __init__(
        self,
        base_model: Transcriber,  # TODO add special code for TimedTranscriber
        additional_model: Transcriber,
    ):
        self.base_model = base_model
        self.additional_model = additional_model
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        base_text = DEFAULT_PARSER.parse_single_variant_transcription(
            self.base_model.transcribe(waveform)
        )
        additional_text = DEFAULT_PARSER.parse_single_variant_transcription(
            self.additional_model.transcribe(waveform)
        )
        matches, _ = solve_optimal_alignment(base_text, additional_text)
        
        skip_chars = set(string.ascii_lowercase + string.digits)
        def shoud_skip(word: str) -> bool:
            return len(skip_chars.intersection(set(word.lower()))) > 0
        
        replacements: list[tuple[Token, str]] = []
        for match in matches.matches:
            if match.status == 'replacement':
                true_word = cast(str, cast(Token, match.true).value)
                pred_word = cast(str, cast(Token, match.pred).value)
                if shoud_skip(true_word) or shoud_skip(pred_word):
                    continue
                if word_freq(pred_word, 'ru') > word_freq(true_word, 'ru'):
                    replacements.append((cast(Token, match.true), pred_word))
        
        replaced_text = base_text.text
        for token, replacement in replacements[::-1]:
            replaced_text = (
                replaced_text[:token.start_pos]
                + replacement
                + replaced_text[token.end_pos:]
            )
        
        return replaced_text