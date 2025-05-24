# nlp_analyzer_service/app/services/analysis_pipeline.py
from loguru import logger
from typing import List, Optional
import numpy as np # For np.max if still used
from datetime import datetime # Ensure datetime is imported for processing_timestamp

from app.models import (
    NLPAnalysisRequest, NLPAnalysisResponse, SentimentScore,
    TopicInfo, KeywordFrequency
)
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
            for i, topic_id_val in enumerate(pred_topic_ids): # pred_topic_ids is List[int]
                topic_keywords_scores = topic_modeler.get_topic_details(topic_id_val) or []
                topic_name_str = topic_modeler.get_topic_name(topic_id_val)
                current_doc_probability: Optional[float] = None

                # Handling probabilities based on BERTopic output format
                if pred_topic_probs_distributions is not None:
                    if isinstance(pred_topic_probs_distributions, np.ndarray) and pred_topic_probs_distributions.ndim == 2:
                        # Case: probabilities is a 2D array (docs x topics) from calculate_probabilities=True
                        # We assume batch of 1, so pred_topic_probs_distributions[0] is the prob distribution for this doc
                        # And topic_id_val is the index (or direct ID) into that distribution
                        try:
                            # If topic_id_val is an index into the full probability array:
                            # current_doc_probability = float(pred_topic_probs_distributions[0, topic_id_val])
                            # However, BERTopic's transform usually gives probability for the *assigned* topic(s).
                            # If pred_topic_ids = [5] and pred_topic_probs_distributions = [0.89] (for that specific topic)
                            if i < len(pred_topic_probs_distributions):
                                prob_val = pred_topic_probs_distributions[i]
                                if isinstance(prob_val, (float, int, np.floating, np.integer)):
                                     current_doc_probability = float(prob_val)
                                elif isinstance(prob_val, np.ndarray) and prob_val.size == 1: # Single value in ndarray
                                     current_doc_probability = float(prob_val.item())

                        except (IndexError, TypeError, ValueError) as e_prob:
                            logger.warning(f"Pipeline: Could not extract probability for topic {topic_id_val}. Data: {pred_topic_probs_distributions}. Error: {e_prob}")
                    elif isinstance(pred_topic_probs_distributions, list) and i < len(pred_topic_probs_distributions):
                        # Case: probabilities is a list, one entry per document in the batch
                        # And each entry could be a float (if single topic assigned) or array (if multiple possible per doc)
                        prob_data_for_doc = pred_topic_probs_distributions[i]
                        try:
                            if isinstance(prob_data_for_doc, (float, int, np.floating, np.integer)): # e.g. [0.89]
                                current_doc_probability = float(prob_data_for_doc)
                            elif isinstance(prob_data_for_doc, np.ndarray) and prob_data_for_doc.size > 0: # e.g. [[0.1, 0.89, ...]]
                                # This might be the full distribution if BERTopic was set to return it this way for transform
                                # For simplicity, if it's an array, take the max as the prob of the *most likely* topic among these.
                                # Or, if topic_id_val is an index, use it. Assume it's the prob for topic_id_val.
                                current_doc_probability = float(np.max(prob_data_for_doc)) # Example: take max if it's a distribution for the doc
                        except (TypeError, ValueError) as e_prob:
                            logger.warning(f"Pipeline: Could not convert probability for topic {topic_id_val}, data: {prob_data_for_doc}. Error: {e_prob}")

                assigned_doc_topics.append(TopicInfo(
                    id=int(topic_id_val), name=topic_name_str, keywords=topic_keywords_scores, probability=current_doc_probability
                ))
        elif request_data.cleaned_text and request_data.cleaned_text.strip(): # Only error if text was not empty
            errors.append("Topic modeling (transform) returned no result for non-empty text.")
    else:
        if not topic_modeler.topic_model:
            errors.append("BERTopic model is not loaded, skipping topic modeling.")
        elif topic_modeler.topic_model and not topic_modeler.topic_model.embedding_model: # Check was for SBERT instance
            errors.append("BERTopic model is loaded, but its SBERT embedding_model is missing. Skipping topic modeling.")

    # 3. Keyword Extraction
    input_tokens_for_freq = request_data.lemmas if request_data.lemmas else request_data.tokens_processed
    freq_keywords_raw = keyword_extractor.extract_from_tokens_frequency(input_tokens_for_freq)
    extracted_keywords_freq = [KeywordFrequency(keyword=kw, frequency=f) for kw, f in freq_keywords_raw]

    sentiment_on_keywords_summary: Optional[List[SentimentScore]] = None
    if extracted_keywords_freq:
        top_overall_keywords_text = " ".join([kf.keyword for kf in extracted_keywords_freq[:keyword_extractor.top_n]])
        if top_overall_keywords_text.strip():
            kw_summary_sentiment_raw = sentiment_analyzer.analyze(top_overall_keywords_text)
            sentiment_on_keywords_summary = [SentimentScore(**s) for s in kw_summary_sentiment_raw] if kw_summary_sentiment_raw else []

    return NLPAnalysisResponse(
        raw_mongo_id=request_data.raw_mongo_id,
        source=request_data.source,
        original_timestamp=request_data.original_timestamp,
        retrieved_by_keyword=request_data.retrieved_by_keyword,
        keyword_concept_id=request_data.keyword_concept_id,             # <<< UPDATED
        original_keyword_language=request_data.keyword_language,       # <<< UPDATED
        processing_timestamp=datetime.utcnow(),                        # Use current time
        detected_language=request_data.detected_language,
        overall_sentiment=overall_sentiment_scores,
        assigned_topics=assigned_doc_topics,
        extracted_keywords_frequency=extracted_keywords_freq,
        sentiment_on_extracted_keywords_summary=sentiment_on_keywords_summary,
        analysis_errors=errors if errors else None
    )