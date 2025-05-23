# nlp_analyzer_service/app/services/analysis_pipeline.py
from loguru import logger
from typing import List, Optional
import numpy as np # For np.max if still used

from app.models import (
    NLPAnalysisRequest, NLPAnalysisResponse, SentimentScore, 
    TopicInfo, KeywordFrequency
)
# Import the actual analyzer classes, not just the getter functions
from app.nlp_tasks.sentiment_analyzer import SentimentAnalyzer
from app.nlp_tasks.topic_modeler import TopicModeler
from app.nlp_tasks.keyword_extractor import KeywordExtractor

async def execute_nlp_pipeline(
    request_data: NLPAnalysisRequest,
    sentiment_analyzer: SentimentAnalyzer,
    topic_modeler: TopicModeler,
    keyword_extractor: KeywordExtractor
) -> NLPAnalysisResponse:
    """
    Executes the full NLP analysis pipeline for a single document request.
    """
    logger.debug(f"Executing NLP pipeline for raw_mongo_id: {request_data.raw_mongo_id}")
    errors: List[str] = []

    # 1. Sentiment Analysis
    overall_sentiment_scores_raw = sentiment_analyzer.analyze(request_data.cleaned_text)
    overall_sentiment_scores = [SentimentScore(**s) for s in overall_sentiment_scores_raw] if overall_sentiment_scores_raw else []
    if not overall_sentiment_scores_raw and request_data.cleaned_text and request_data.cleaned_text.strip():
        errors.append("Overall sentiment analysis failed or returned no result for non-empty text.")

    # 2. Topic Modeling
    assigned_doc_topics: List[TopicInfo] = []
    if topic_modeler.topic_model and topic_modeler.topic_model.embedding_model:
        topic_result = topic_modeler.get_topics_for_doc(request_data.cleaned_text)
        if topic_result:
            pred_topic_ids, pred_topic_probs_distributions = topic_result
            for i, topic_id_val in enumerate(pred_topic_ids):
                topic_keywords_scores = topic_modeler.get_topic_details(topic_id_val) or []
                topic_name_str = topic_modeler.get_topic_name(topic_id_val)
                current_doc_probability: Optional[float] = None
                if pred_topic_probs_distributions is not None and \
                   i < len(pred_topic_probs_distributions) and \
                   pred_topic_probs_distributions[i] is not None:
                    prob_data_for_doc = pred_topic_probs_distributions[i]
                    try:
                        if isinstance(prob_data_for_doc, np.ndarray) and prob_data_for_doc.ndim > 0:
                            current_doc_probability = float(np.max(prob_data_for_doc))
                        elif isinstance(prob_data_for_doc, (float, int, np.floating, np.integer)):
                            current_doc_probability = float(prob_data_for_doc)
                        # else: logger.warning for unexpected type already in TopicModeler
                    except (TypeError, ValueError) as e_prob:
                        logger.warning(f"Pipeline: Could not convert probability for topic {topic_id_val}, data: {prob_data_for_doc}. Error: {e_prob}")
                assigned_doc_topics.append(TopicInfo(
                    id=int(topic_id_val), name=topic_name_str, keywords=topic_keywords_scores, probability=current_doc_probability
                ))
        elif request_data.cleaned_text and request_data.cleaned_text.strip():
            errors.append("Topic modeling (transform) returned no result for non-empty text.")
    else:
        if not topic_modeler.topic_model:
            errors.append("BERTopic model is not loaded, skipping topic modeling.")
        elif topic_modeler.topic_model and not topic_modeler.topic_model.embedding_model: # Check added
            errors.append("BERTopic model is loaded, but its SBERT embedding_model is missing. Skipping topic modeling.")

    # 3. Keyword Extraction
    input_tokens_for_freq = request_data.lemmas if request_data.lemmas else request_data.tokens_processed
    freq_keywords_raw = keyword_extractor.extract_from_tokens_frequency(input_tokens_for_freq)
    extracted_keywords_freq = [KeywordFrequency(keyword=kw, frequency=f) for kw, f in freq_keywords_raw]

    sentiment_on_keywords_summary: Optional[List[SentimentScore]] = None
    if extracted_keywords_freq:
        top_overall_keywords_text = " ".join([kf.keyword for kf in extracted_keywords_freq[:keyword_extractor.top_n]]) # Use keyword_extractor.top_n
        if top_overall_keywords_text.strip():
            kw_summary_sentiment_raw = sentiment_analyzer.analyze(top_overall_keywords_text)
            sentiment_on_keywords_summary = [SentimentScore(**s) for s in kw_summary_sentiment_raw] if kw_summary_sentiment_raw else []

    return NLPAnalysisResponse(
        raw_mongo_id=request_data.raw_mongo_id,
        source=request_data.source,
        original_timestamp=request_data.original_timestamp,
        retrieved_by_keyword=request_data.retrieved_by_keyword,
        detected_language=request_data.detected_language,
        overall_sentiment=overall_sentiment_scores,
        assigned_topics=assigned_doc_topics,
        extracted_keywords_frequency=extracted_keywords_freq,
        sentiment_on_extracted_keywords_summary=sentiment_on_keywords_summary,
        analysis_errors=errors if errors else None
    )