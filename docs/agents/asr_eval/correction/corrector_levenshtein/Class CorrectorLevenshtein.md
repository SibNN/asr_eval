# Class CorrectorLevenshtein (defined in asr_eval/correction/corrector_levenshtein.py at lines 41-124)

@dataclasses.dataclass
class CorrectorLevenshtein(asr_eval.correction.interfaces.TranscriptionCorrector):
    """Finds rare words in the transcription, searches for similar words
    in the :code:`domain_specific_bag_of_words` corpus, replaces if
    found, inflects accordingly.

    Works for Russian language currently.

    Author: Yana Fitkovskaja; Updated by: Oleg Sedukhin
    """
    ...

    domain_specific_bag_of_words: list[str]

    freq_threshold: float = 1

    distance_thresholds: list[float] = field(
        default_factory=lambda: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    )

    @typing.override
    def correct(
        self, transcription: str, waveform: asr_eval.utils.types.FLOATS | None = None
    ) -> str:
    ...

    def get_word_corrections(
        self, transcription: str
    ) -> list[asr_eval.correction.corrector_levenshtein.WordCorrection]:
    ...