# nlp_analyzer_service/app/nlp_tasks/keyword_extractor.py
from collections import Counter
from typing import List, Tuple, Optional
from loguru import logger
import yake
from app.config import settings # CORRECTED IMPORT

class KeywordExtractor:
    def __init__(self, top_n: int = settings.TOP_N_KEYWORDS): # CORRECTED DEFAULT
        self.top_n = top_n

    def extract_from_text(self, text: str, language: Optional[str] = "en") -> List[Tuple[str, float]]:
        if not text or not text.strip(): logger.warning("Input text for keyword extraction is empty."); return []
        try:
            valid_yake_lang = language if language and len(language) == 2 else "en"
            custom_kw_extractor = yake.KeywordExtractor(lan=valid_yake_lang, n=3, dedupLim=0.9, top=self.top_n, features=None)
            return custom_kw_extractor.extract_keywords(text)
        except Exception as e: logger.error(f"Error during YAKE keyword extraction (lang: {language}): {e}", exc_info=True); return []

    def extract_from_tokens_frequency(self, processed_tokens: Optional[List[str]]) -> List[Tuple[str, int]]:
        if not processed_tokens: return []
        try: return Counter(processed_tokens).most_common(self.top_n)
        except Exception as e: logger.error(f"Error during frequency-based keyword extraction: {e}", exc_info=True); return []

keyword_extractor_instance = KeywordExtractor()
def get_keyword_extractor_instance(): return keyword_extractor_instance