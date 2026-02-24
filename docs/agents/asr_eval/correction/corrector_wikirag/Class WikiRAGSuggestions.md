# Class WikiRAGSuggestions (defined in asr_eval/correction/corrector_wikirag.py at lines 55-69)

@dataclasses.dataclass
class WikiRAGSuggestions:
    """
    A list of suggestions returned by
    :class:`~asr_eval.correction.corrector_wikirag.WikipediaTermRetriever`.

    Work in progress.
    """
    ...

    original_text: str

    detected_topic: str

    query_terms: list[str]

    suggested_terms: list[str]

    term_scores: list[float]