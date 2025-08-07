from dataclasses import dataclass
import re
from collections import Counter
import sys
from typing import cast, override

import wikipediaapi
import requests_cache
from transformers import pipeline # type: ignore
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity # type: ignore
import nltk
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords
from tqdm.auto import tqdm

from ..utils.types import FLOATS
from .interfaces import TranscriptionCorrector


__all__ = [
    'TOPICS',
    'WikiArticle',
    'WikiRAGSuggestions',
    'WikipediaTermRetriever',
]


def _vectors_cosine_similarity(vec1: FLOATS, vec2: FLOATS) -> float:
    return _cosine_similarity(vec1[None], vec2[None])[0, 0]
    


def _download_nltk_resources():
    resources = ['punkt', 'stopwords', 'perluniprops', 'nonbreaking_prefixes', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')  # type: ignore
        except LookupError:
            nltk.download(resource, quiet=True)  # type: ignore


TOPICS = [
    "биология", "химия", "физика", "история", "медицина", "география", "искусство", "литература"
]


@dataclass
class WikiArticle:
    '''
    A Wikipedia page for RAG purposes.
    '''
    title: str
    text: str
    url: str


@dataclass
class WikiRAGSuggestions:
    '''
    A list of suggestions returned by `WikipediaTermRetriever`.
    
    Work in progress.
    '''
    original_text: str
    detected_topic: str
    query_terms: list[str]
    suggested_terms: list[str]
    term_scores: list[float]


class WikipediaTermRetriever(TranscriptionCorrector):
    '''
    A term retriever capable of correcting transcriptions.
    
    Work in progress.
    
    Author: Timur Rafikov
    Updated by: Oleg Sedukhin
    '''
    def __init__(
        self,
        lang: str = "ru",
        candidate_topics: list[str] = TOPICS,
        score_threshold: float = 0.7,
        verbose: bool = False,
    ):
        _download_nltk_resources()
        
        self.wiki = wikipediaapi.Wikipedia("MyRAGASR", lang)
        
        self.wiki._session = requests_cache.CachedSession( # pyright: ignore[reportPrivateUsage]
            # not working
            'tmp/wikipedia_cache', backend='sqlite',
        )
        if verbose:  
            wikipediaapi.log.setLevel(level=wikipediaapi.logging.DEBUG)
            out_hdlr = wikipediaapi.logging.StreamHandler(sys.stderr)
            out_hdlr.setFormatter(wikipediaapi.logging.Formatter('%(asctime)s %(message)s'))
            out_hdlr.setLevel(wikipediaapi.logging.DEBUG)
            wikipediaapi.log.addHandler(out_hdlr)
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="vicgalle/xlm-roberta-large-xnli-anli"
        )
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.stopwords = set(stopwords.words("russian" if lang == 'ru' else 'english')) # type: ignore
        self.candidate_topics = candidate_topics
        self.retrieved_articles: dict[str, list[WikiArticle]] = {}
        self.score_threshold = score_threshold
        
        
    @override
    def correct(self, transcription: str, waveform: FLOATS | None = None) -> str:
        suggestions = self.process_query(transcription, top_terms=10)
        
        for query_term, suggested_term, score in zip(
            suggestions.query_terms,
            suggestions.suggested_terms,
            suggestions.term_scores,
            strict=True,
        ):
            assert query_term in transcription
            if score > self.score_threshold:
                transcription = transcription.replace(query_term, suggested_term)
               
        return transcription


    def detect_topic(self, text: str) -> str:
        """Определение темы с помощью zero-shot классификации"""
        result = self.classifier(text, self.candidate_topics) # type: ignore
        return result['labels'][0] # type: ignore


    SKIP_PAGE_TITLE_SUBSTRINGS = [
        "Категория:Википедисты",
        "Категория:Разделы Википедии",
        "Категория:Википедия:Хорошие статьи",
    ]


    def get_category_articles(self, category_name: str, max_articles: int = 500) -> list[WikiArticle]:
        """Рекурсивная загрузка статей категории"""
        if category_name in self.retrieved_articles:
            return self.retrieved_articles[category_name]
        
        category = self.wiki.page(f"Category:{category_name}")
        assert category.exists()
            
        articles: list[WikiArticle] = []
        
        for page in (pbar := tqdm(category.categorymembers.values())):
            pbar.set_description(page.title)
            if any(
                (substr.lower() in page.title.lower())
                for substr in self.SKIP_PAGE_TITLE_SUBSTRINGS
            ):
                continue
            print(page.title)
            if len(articles) > min(5000, max_articles):
                break
            if page.ns == wikipediaapi.Namespace.MAIN:
                articles.append(WikiArticle(
                    title=page.title,
                    text=page.text[:10000],  # Ограничиваем размер
                    url=str(page.fullurl),
                ))
            elif page.ns == wikipediaapi.Namespace.CATEGORY:
                sub_articles = self.get_category_articles(page.title.split(":")[1])
                articles.extend(sub_articles[:max_articles - len(articles)])
        
        self.retrieved_articles[category_name] = articles
        return articles


    def text_to_terms(self, text: str) -> list[str]:
        """Токенизация и очистка текста"""
        tokens = word_tokenize(text.lower())
        return [
            token for token in tokens 
            if re.fullmatch(r'[а-яё]{3,}', token) and token not in self.stopwords
        ]


    def build_term_index(self, articles: list[WikiArticle]) -> dict[str, FLOATS]:
        """Создание семантического индекса терминов"""
        all_terms: list[str] = []
        for article in articles:
            all_terms += self.text_to_terms(article.text)
        unique_terms = [term for term, count in Counter(all_terms).items() if count < 20]
        
        # Создаем эмбеддинги для терминов
        term_embeddings: dict[str, FLOATS] = {}
        batch_size = 100
        for i in range(0, len(unique_terms), batch_size):
            batch = unique_terms[i:i+batch_size]
            embeddings = cast(FLOATS, self.embedder.encode(batch)) # type: ignore
            for term, emb in zip(batch, embeddings):
                term_embeddings[term] = emb
                
        return term_embeddings


    def find_similar_terms(
        self,
        query_terms: list[str],
        term_index: dict[str, FLOATS],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> dict[str, list[tuple[str, float]]]:
        """Поиск семантически похожих терминов с учетом возможных ошибок"""
        similar_terms: dict[str, list[tuple[str, float]]] = {}
        
        # Получаем все термины из индекса
        all_index_terms = list(term_index.keys())
        
        for term in query_terms:
            # Если термин есть в индексе - используем обычный поиск
            if term in term_index:
                term_emb = term_index[term]
                similarities = {
                    other_term: _vectors_cosine_similarity(term_emb, other_emb)
                    for other_term, other_emb in term_index.items()
                    if term != other_term
                }
                    
                sorted_terms = sorted(similarities.items(), key=lambda x: -x[1])[:top_k]
                similar_terms[term] = sorted_terms
            else:
                # Если термина нет в индексе - ищем похожие по написанию и смыслу
                term_emb = cast(FLOATS, self.embedder.encode(term)) # type: ignore
                candidate_terms: list[tuple[str, float]] = []
                
                # Этап 1: Быстрый поиск по сходству строк (для опечаток)
                for index_term in all_index_terms:
                    # Используем комбинацию семантического и строкового сходства
                    semantic_sim = _vectors_cosine_similarity(term_emb, term_index[index_term])
                    fuzzy_sim = fuzz.ratio(term, index_term) / 100
                    combined_score = 0.6 * semantic_sim + 0.4 * fuzzy_sim
                    
                    if combined_score > similarity_threshold:
                        candidate_terms.append((index_term, combined_score))
                
                # Этап 2: Выбираем лучшие кандидаты
                if candidate_terms:
                    candidate_terms.sort(key=lambda x: -x[1])
                    similar_terms[term] = candidate_terms[:top_k]
                else:
                    similar_terms[term] = []
                
        return similar_terms


    def process_query(self, asr_text: str, top_terms: int = 10):
        """Полный цикл обработки запроса"""
        # 1. Определяем тему
        topic = self.detect_topic(asr_text)
        print(f"Определена тема: {topic}")
        
        # 2. Загружаем статьи по теме
        articles = self.get_category_articles(topic)
        print(f"Загружено статей: {len(articles)}")
        
        # 3. Строим семантический индекс терминов
        term_index = self.build_term_index(articles)
        print(f"Проиндексировано терминов: {len(term_index)}")
        
        # 4. Извлекаем термины из запроса
        query_terms = self.text_to_terms(asr_text)
        print(f"Термины для коррекции: {query_terms}")
        
        # 5. Находим похожие термины
        similar_terms = self.find_similar_terms(query_terms, term_index)
        print(f"Найденные похожие термины: {similar_terms}")
        
        # 6. Выбираем топ-N терминов
        all_terms: list[tuple[str, float]] = []
        for _term, suggestions in similar_terms.items():
            for suggested_term, score in suggestions:
                all_terms.append((suggested_term, score))
                
        top_terms_list = sorted(all_terms, key=lambda x: -x[1])[:top_terms]
        
        return WikiRAGSuggestions(
            original_text=asr_text,
            detected_topic=topic,
            query_terms=query_terms,
            suggested_terms=[term[0] for term in top_terms_list],
            term_scores=[term[1] for term in top_terms_list],
        )