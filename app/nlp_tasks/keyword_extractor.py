# Keyword Extractor Logic
from collections import Counter
from typing import List, Tuple, Optional
from loguru import logger
import yake

from app.config import TOP_N_KEYWORDS

class KeywordExtractor:
    def __init__(self, top_n: int = TOP_N_KEYWORDS):
        self.top_n = top_n
        self.kw_extractor = yake.KeywordExtractor(n=3, dedupLim=0.9, top=top_n, features=None)

    def extract_from_text(self, text: str, language: Optional[str] = "en") -> List[Tuple[str, float]]:
        if not text or not text.strip():
            logger.warning("Input text for keyword extraction is empty.")
            return []
        try:
            custom_kw_extractor = yake.KeywordExtractor(lan=language if language else "en", 
                                                        n=3, dedupLim=0.9, top=self.top_n, features=None)
            keywords_with_scores = custom_kw_extractor.extract_keywords(text)
            return keywords_with_scores
        except Exception as e:
            logger.error(f"Error during YAKE keyword extraction: {e}", exc_info=True)
            return []

    def extract_from_tokens_frequency(self, processed_tokens: Optional[List[str]]) -> List[Tuple[str, int]]:
        if not processed_tokens:
            return []
        try:
            counts = Counter(processed_tokens)
            top_keywords = counts.most_common(self.top_n)
            return top_keywords
        except Exception as e:
            logger.error(f"Error during frequency-based keyword extraction: {e}", exc_info=True)
            return []

keyword_extractor_instance = KeywordExtractor()

def get_keyword_extractor_instance():
    return keyword_extractor_instance