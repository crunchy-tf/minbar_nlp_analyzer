# nlp_analyzer_service/app/models.py
# Pydantic models for request/response
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class NLPAnalysisRequest(BaseModel):
    raw_mongo_id: str = Field(..., example="680a75cf622ece5a1dc7a4bc")
    source: str = Field(..., example="post")
    original_timestamp: datetime = Field(..., example="2025-04-16T19:20:19Z")
    retrieved_by_keyword: str = Field(..., example="تكاليف الأدوية")
    keyword_language: str = Field(..., example="ar") # This IS the original keyword language
    keyword_concept_id: Optional[str] = Field(None, description="Concept ID from Keyword Manager that sourced the keyword") # <<< UPDATED/ADDED

    detected_language: Optional[str] = Field(None, example="ar")
    cleaned_text: str = Field(..., example="في إطار زيارته الحالية لدولة تونس وزير الاستثمار...")
    tokens_processed: Optional[List[str]] = Field(None, example=["في", "إطار", "..."])
    lemmas: Optional[List[str]] = Field(None, example=["فِي", "إِطَار", "..."])
    original_url: Optional[str] = Field(None, example="http://example.com/article")


class SentimentScore(BaseModel):
    label: str
    score: float

class TopicInfo(BaseModel):
    id: int
    name: str
    keywords: List[tuple[str, float]] # BERTopic's get_topic returns list of (keyword, score) tuples
    probability: Optional[float] = None

class KeywordFrequency(BaseModel):
    keyword: str
    frequency: int

class NLPAnalysisResponse(BaseModel):
    raw_mongo_id: str
    source: str
    original_timestamp: datetime
    retrieved_by_keyword: str
    keyword_concept_id: Optional[str] = Field(None, description="Concept ID from Keyword Manager that sourced the keyword") # <<< ADDED
    original_keyword_language: Optional[str] = Field(None, description="Language of the original keyword used for ingestion") # <<< ADDED
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)

    detected_language: Optional[str]
    overall_sentiment: List[SentimentScore]
    assigned_topics: List[TopicInfo] # This will store a list of TopicInfo objects

    extracted_keywords_frequency: List[KeywordFrequency]
    sentiment_on_extracted_keywords_summary: Optional[List[SentimentScore]] = None

    analysis_errors: Optional[List[str]] = None