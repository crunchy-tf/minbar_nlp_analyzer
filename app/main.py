# nlp_analyzer_service/app/main.py
from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from datetime import datetime
from typing import List, Optional
import sys
import asyncio # Add asyncio
from contextlib import asynccontextmanager # Add asynccontextmanager

# Updated imports for Pydantic settings and new modules
from app.config import settings # Use Pydantic settings object
from app.models import (
    NLPAnalysisRequest, NLPAnalysisResponse, SentimentScore, 
    TopicInfo, KeywordFrequency
)
from app.nlp_tasks.sentiment_analyzer import get_sentiment_analyzer_instance, SentimentAnalyzer
from app.nlp_tasks.topic_modeler import get_topic_modeler_instance, TopicModeler
from app.nlp_tasks.keyword_extractor import get_keyword_extractor_instance, KeywordExtractor
# Import new modules
from app.db_connector.pg_connector import connect_db as connect_pg, close_db as close_pg, store_nlp_analysis_results # For storing API results
from app.services.scheduler_service import start_scheduler, stop_scheduler
# Main processor job for scheduled task (not directly used by API endpoint, but by scheduler)
from app.main_processor import scheduled_nlp_job 

logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL.upper()) # Use settings for log level

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{settings.SERVICE_NAME} starting up...")
    global sentiment_analyzer_global, topic_modeler_global, keyword_extractor_global # To hold instances

    # 1. Load NLP Models
    try:
        sentiment_analyzer_global = get_sentiment_analyzer_instance()
        topic_modeler_global = get_topic_modeler_instance()
        keyword_extractor_global = get_keyword_extractor_instance()

        if not sentiment_analyzer_global.classifier:
            raise RuntimeError("Sentiment Analyzer model failed to load.")
        if not topic_modeler_global.topic_model:
            raise RuntimeError(f"BERTopic model at {settings.BERTOPIC_MODEL_PATH} could not be loaded.")
        if topic_modeler_global.topic_model and not topic_modeler_global.topic_model.embedding_model:
            raise RuntimeError("BERTopic model loaded, but its SBERT embedding_model is missing.")
        logger.info("NLP models initialized successfully.")
    except Exception as e:
        logger.critical(f"Critical error during NLP model initialization: {e}", exc_info=True)
        raise RuntimeError(f"NLP model initialization failed: {e}") from e

    # 2. Connect to Database
    try:
        await connect_pg() # Connect to PostgreSQL
    except Exception as e:
        logger.critical(f"Database connection failed during startup: {e}", exc_info=True)
        await close_pg() # Attempt to clean up if pool was partially created
        raise RuntimeError(f"Database connection failed: {e}") from e
    
    # 3. Start APScheduler
    try:
        await start_scheduler() # This will add and start the scheduled_nlp_job
    except Exception as e:
        logger.error(f"APScheduler failed to start: {e}", exc_info=True)
        # Decide if this is fatal. For now, we'll let the app start but log error.
        # If scheduler is critical, re-raise a RuntimeError here.

    logger.info(f"{settings.SERVICE_NAME} startup complete.")
    
    yield # Application runs
    
    logger.info(f"{settings.SERVICE_NAME} shutting down...")
    await stop_scheduler()
    await close_pg()
    logger.info("APScheduler and DB connection shut down.")
    logger.info(f"{settings.SERVICE_NAME} shutdown complete.")

app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Microservice for sentiment analysis, topic modeling, and keyword extraction.",
    version="1.0.0",
    lifespan=lifespan # Use the new lifespan manager
)

# No need for global scheduler instance here anymore, it's in scheduler_service
# No need for global NLP instances here if using Depends correctly for the API,
# but the main_processor needs them, so they are initialized in lifespan.

# The log_heartbeat job is now part of the main scheduled_nlp_job or can be a separate job in scheduler_service.py
# For simplicity, we'll assume the main job's logging is sufficient indication of being alive.

@app.post("/analyze", response_model=NLPAnalysisResponse)
async def analyze_text_endpoint( # Renamed to avoid conflict with main_processor's potential helper
    request_data: NLPAnalysisRequest,
    # These Depends will use the globally initialized instances from startup
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer_instance),
    topic_modeler: TopicModeler = Depends(get_topic_modeler_instance),
    keyword_extractor: KeywordExtractor = Depends(get_keyword_extractor_instance)
):
    logger.info(f"API /analyze request for raw_mongo_id: {request_data.raw_mongo_id}")
    
    # --- Replicate core analysis logic here or call a shared function ---
    # This is the same logic as in _perform_single_doc_analysis from main_processor.py
    # For DRY principle, this logic should be in a shared utility/service function.
    # For this example, I'll paste it, but ideally, refactor.

    errors = []
    overall_sentiment_scores_raw = sentiment_analyzer.analyze(request_data.cleaned_text)
    overall_sentiment_scores = [SentimentScore(**s) for s in overall_sentiment_scores_raw] if overall_sentiment_scores_raw else []
    if not overall_sentiment_scores_raw and request_data.cleaned_text.strip():
        errors.append("Overall sentiment analysis failed or returned no result.")

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
                        if isinstance(prob_data_for_doc, np.ndarray) and prob_data_for_doc.ndim > 0 :
                            current_doc_probability = float(np.max(prob_data_for_doc))
                        elif isinstance(prob_data_for_doc, (float, int, np.floating, np.integer)):
                            current_doc_probability = float(prob_data_for_doc)
                    except (TypeError, ValueError):
                        pass # Error already logged by TopicModeler potentially
                assigned_doc_topics.append(TopicInfo(
                    id=int(topic_id_val), name=topic_name_str, keywords=topic_keywords_scores, probability=current_doc_probability
                ))
        elif request_data.cleaned_text.strip(): errors.append("Topic modeling returned no result.")
    else:
        if not topic_modeler.topic_model: errors.append("BERTopic model is not loaded.")
        elif not topic_modeler.topic_model.embedding_model: errors.append("BERTopic SBERT embedding_model is missing.")

    input_tokens_for_freq = request_data.lemmas if request_data.lemmas else request_data.tokens_processed
    freq_keywords_raw = keyword_extractor.extract_from_tokens_frequency(input_tokens_for_freq)
    extracted_keywords_freq = [KeywordFrequency(keyword=kw, frequency=f) for kw, f in freq_keywords_raw]

    sentiment_on_keywords_summary: Optional[List[SentimentScore]] = None
    if extracted_keywords_freq:
        top_overall_keywords_text = " ".join([kf.keyword for kf in extracted_keywords_freq[:keyword_extractor.top_n]])
        if top_overall_keywords_text.strip():
            kw_summary_sentiment_raw = sentiment_analyzer.analyze(top_overall_keywords_text)
            sentiment_on_keywords_summary = [SentimentScore(**s) for s in kw_summary_sentiment_raw] if kw_summary_sentiment_raw else []

    # --- End Replicated Logic ---

    nlp_response = NLPAnalysisResponse(
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

    # Store the result from the API call as well
    try:
        await store_nlp_analysis_results([nlp_response])
        logger.info(f"API call result for {request_data.raw_mongo_id} stored in DB.")
    except Exception as e:
        logger.error(f"Failed to store API call result for {request_data.raw_mongo_id} in DB: {e}", exc_info=True)
        # Decide if this should be a 500 error for the API caller
        # For now, the analysis itself succeeded, so we still return 200
        # but log the storage failure.

    return nlp_response

if __name__ == "__main__":
    import uvicorn
    # Note: SBERT_MODEL_NAME is now loaded from settings object, not global config directly.
    # The TopicModeler init itself will log if settings.SBERT_MODEL_NAME is missing.
    logger.info("Starting NLP Analyzer Service for local development...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.TARGET_POSTGRES_PORT if settings.TARGET_POSTGRES_PORT else 8001, log_level=settings.LOG_LEVEL.lower(), reload=True) # Use a different port or make it configurable if 8001 is taken