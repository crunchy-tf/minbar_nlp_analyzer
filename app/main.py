# FILE: app/main.py
# FastAPI app, routes, scheduler
from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from datetime import datetime
from typing import List, Optional
import sys
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import numpy as np # Import numpy

from app.models import (
    NLPAnalysisRequest, NLPAnalysisResponse, SentimentScore, 
    TopicInfo, KeywordFrequency
)
from app.config import LOG_LEVEL, SCHEDULER_JOB_INTERVAL_SECONDS, BERTOPIC_MODEL_PATH
from app.nlp_tasks.sentiment_analyzer import get_sentiment_analyzer_instance, SentimentAnalyzer
from app.nlp_tasks.topic_modeler import get_topic_modeler_instance, TopicModeler
from app.nlp_tasks.keyword_extractor import get_keyword_extractor_instance, KeywordExtractor

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

app = FastAPI(
    title="Minbar NLP Analyzer Service",
    description="Microservice for sentiment analysis, topic modeling, and keyword extraction.",
    version="1.0.0"
)

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    logger.info("NLP Analyzer Service starting up...")
    get_sentiment_analyzer_instance() 
    tm_instance = get_topic_modeler_instance() 
    if not tm_instance.topic_model:
        logger.critical(f"BERTopic model at {BERTOPIC_MODEL_PATH} could not be loaded or its embedding model is missing. Topic modeling in /analyze will be skipped or fail.")
    elif tm_instance.topic_model and not tm_instance.topic_model.embedding_model:
         logger.critical(f"BERTopic model at {BERTOPIC_MODEL_PATH} loaded, but its internal SBERT embedding_model is missing. Topic modeling in /analyze will fail.")
    get_keyword_extractor_instance() 
    
    scheduler.add_job(
        log_heartbeat,
        trigger=IntervalTrigger(seconds=SCHEDULER_JOB_INTERVAL_SECONDS),
        id="heartbeat_job",
        name="Log service heartbeat",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started.")
    logger.info("NLP Analyzer Service startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("NLP Analyzer Service shutting down...")
    if scheduler.running:
        scheduler.shutdown()
    logger.info("APScheduler shut down.")
    logger.info("NLP Analyzer Service shutdown complete.")

async def log_heartbeat():
    logger.info(f"NLP Analyzer Service is alive. Last heartbeat: {datetime.utcnow()}")


@app.post("/analyze", response_model=NLPAnalysisResponse)
async def analyze_text(
    request_data: NLPAnalysisRequest,
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer_instance),
    topic_modeler: TopicModeler = Depends(get_topic_modeler_instance),
    keyword_extractor: KeywordExtractor = Depends(get_keyword_extractor_instance)
):
    logger.info(f"Received analysis request for raw_mongo_id: {request_data.raw_mongo_id}")
    errors = []

    # 1. Zero-Shot Sentiment Analysis on the whole cleaned_text
    overall_sentiment_scores_raw = sentiment_analyzer.analyze(request_data.cleaned_text)
    overall_sentiment_scores = [SentimentScore(**s) for s in overall_sentiment_scores_raw] if overall_sentiment_scores_raw else []
    if not overall_sentiment_scores_raw and request_data.cleaned_text.strip():
        errors.append("Overall sentiment analysis failed or returned no result.")


    # 2. Topic Modeling
    assigned_doc_topics: List[TopicInfo] = []

    if topic_modeler.topic_model and topic_modeler.topic_model.embedding_model:
        topic_result = topic_modeler.get_topics_for_doc(request_data.cleaned_text) 
        
        if topic_result:
            pred_topic_ids, pred_topic_probs_distributions = topic_result

            # For a single input document like [doc_text] to BERTopic's transform,
            # pred_topic_ids is a list like [assigned_topic_id_for_doc_0]
            # pred_topic_probs_distributions is a list containing one numpy array:
            #   [array_of_all_topic_probs_for_doc_0]
            # or it could be just the single probability for the assigned topic if not full distribution.
            
            for i, topic_id in enumerate(pred_topic_ids): # i will typically be 0 for single doc processing
                topic_keywords_scores = topic_modeler.get_topic_details(topic_id) or []
                topic_name_str = topic_modeler.get_topic_name(topic_id)
                
                current_doc_probability: Optional[float] = None
                if pred_topic_probs_distributions is not None and \
                   i < len(pred_topic_probs_distributions) and \
                   pred_topic_probs_distributions[i] is not None:
                    
                    prob_data_for_doc = pred_topic_probs_distributions[i]
                    
                    try:
                        if isinstance(prob_data_for_doc, np.ndarray) and prob_data_for_doc.ndim > 0 :
                            # If it's an array, it's likely the full distribution for this document.
                            # The assigned topic_id is the one with the max probability.
                            current_doc_probability = float(np.max(prob_data_for_doc))
                            logger.debug(f"Extracted max probability {current_doc_probability} from distribution for topic {topic_id}")
                        elif isinstance(prob_data_for_doc, (float, int, np.floating, np.integer)):
                            # If it's already a scalar, BERTopic returned the single highest probability.
                            current_doc_probability = float(prob_data_for_doc)
                            logger.debug(f"Using direct probability {current_doc_probability} for topic {topic_id}")
                        else:
                            logger.warning(f"Unexpected type for probability data for doc index {i}, topic {topic_id}. Type: {type(prob_data_for_doc)}. Value: {prob_data_for_doc}")
                    except (TypeError, ValueError) as e_prob:
                        logger.warning(f"Could not convert probability to float for doc index {i}, topic {topic_id}. Prob data: {prob_data_for_doc}. Error: {e_prob}")
                
                assigned_doc_topics.append(TopicInfo(
                    id=int(topic_id),
                    name=topic_name_str,
                    keywords=topic_keywords_scores,
                    probability=current_doc_probability
                ))
        elif request_data.cleaned_text.strip():
            errors.append("Topic modeling (transform) returned no result for non-empty text.")
            
    else:
        if not topic_modeler.topic_model:
            errors.append("BERTopic model is not loaded, skipping topic modeling.")
        elif not topic_modeler.topic_model.embedding_model:
            errors.append("BERTopic model is loaded, but its SBERT embedding_model is missing. Skipping topic modeling.")


    # 3. Keyword Extraction (Frequency-based on preprocessed lemmas/tokens)
    input_tokens_for_freq = request_data.lemmas if request_data.lemmas else request_data.tokens_processed
    
    freq_keywords_raw = keyword_extractor.extract_from_tokens_frequency(input_tokens_for_freq)
    extracted_keywords_freq = [KeywordFrequency(keyword=kw, frequency=f) for kw, f in freq_keywords_raw]

    # Sentiment on summary of extracted keywords
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
        
        detected_language=request_data.detected_language,
        overall_sentiment=overall_sentiment_scores,
        assigned_topics=assigned_doc_topics,
        extracted_keywords_frequency=extracted_keywords_freq,
        sentiment_on_extracted_keywords_summary=sentiment_on_keywords_summary,
        analysis_errors=errors if errors else None
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NLP Analyzer Service for local development...")
    from app.config import SBERT_MODEL_NAME 
    if not SBERT_MODEL_NAME:
        logger.error("SBERT_MODEL_NAME not found in app.config.py! TopicModeler may fail.")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")