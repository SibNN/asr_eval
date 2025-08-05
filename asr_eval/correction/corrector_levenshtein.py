from dataclasses import dataclass
import re
from typing import cast, override

import Levenshtein
import numpy as np

from .interfaces import TranscriptionCorrector
from ..linguistics.linguistics import lemmatize_ru, try_inflect_ru, word_freq


@dataclass
class WordCorrection:
    start: int
    end: int
    correction: str


def apply_corrections(text: str, corrections: list[WordCorrection]) -> str:
    corrections = sorted(corrections, key=lambda c: c.start)
    for c in corrections[::-1]:
        text = text[:c.start] + c.correction + text[c.end:]
    return text


@dataclass
class CorrectorLevenshtein(TranscriptionCorrector):
    '''
    Finds rare words in the transcription, searches for similar words in the
    `domain_specific_bag_of_words` corpus, replaces if found and inflects accordingly.
    
    Author: Yana Fitkovskaya
    Updated by: Oleg Sedukhin
    '''
    domain_specific_bag_of_words: list[str]
    freq_threshold: float = 1
    distance_thresholds: list[float] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    
    @override
    def correct(self, transcription: str) -> str:
        corrections = self.get_word_corrections(transcription)
        return apply_corrections(transcription, corrections)
    
    def get_word_corrections(self, transcription: str) -> list[WordCorrection]:
        corrections: list[WordCorrection] = []
        
        for match in re.finditer(r'\w+', transcription):
            orig_word = match.group()
            
            distance_threshold = (
                self.distance_thresholds[len(orig_word)]
                if len(orig_word) < len(self.distance_thresholds)
                else self.distance_thresholds[-1]
            )
            if distance_threshold == 0:
                continue
            
            lemmatized = lemmatize_ru(_preprocess_text(orig_word))
            
            if word_freq(lemmatized, lang='ru') > self.freq_threshold:
                continue
            
            distances = [
                Levenshtein.distance(lemmatized, _preprocess_text(word))
                for word in self.domain_specific_bag_of_words
            ]
            min_distance = min(distances)
            if min_distance > distance_threshold:
                continue
            
            indices = np.where(np.array(distances) == min_distance)[0]
            if len(indices) > 1:
                continue
            
            closest_word = self.domain_specific_bag_of_words[cast(int, indices[0])]
            closest_word_inflected, inflect_status = try_inflect_ru(closest_word, orig_word)
            print(closest_word_inflected, inflect_status)
            
            if inflect_status == 'fail':
                continue
            
            if orig_word[0].isupper():
                closest_word_inflected = closest_word_inflected.capitalize()
            
            corrections.append(WordCorrection(
                match.start(),
                match.end(),
                closest_word_inflected,
            ))
        
        return corrections


def _preprocess_text(word: str) -> str:
    '''
    Preprocesses a word so that levenshtein distance between words reflects their
    phonetic similarity better.
    
    The current implementation also replaces "ё" with "е" because in many texts "ё"
    is not used.
    '''
    return word.lower().replace('ё', 'е')