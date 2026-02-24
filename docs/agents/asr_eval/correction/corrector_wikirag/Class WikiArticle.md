# Class WikiArticle (defined in asr_eval/correction/corrector_wikirag.py at lines 46-53)

@dataclasses.dataclass
class WikiArticle:
    """A Wikipedia page for RAG purposes."""
    ...

    title: str

    text: str

    url: str