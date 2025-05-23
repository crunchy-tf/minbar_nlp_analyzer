# nlp_analyzer_service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, status # Added status
from loguru import logger
from datetime import datetime
from typing import List, Optional
import sys
import asyncio 
from contextlib import asynccontextmanager 
import numpy as np # Keep for any numpy usage in pipeline if it creeps in

from app.config import settings 
from app.models import (
    NLPAnalysisRequest, NLPAnalysisResponse, SentimentScore, 
    TopicInfo, KeywordFrequency # Ensure all Pydantic models are imported
)
from app.nlp_tasks.sentiment_analyzer import get_sentiment_analyzer_instance, SentimentAnalyzer
from app.nlp_tasks.topic_modeler import get_topic_modeler_instance, TopicModeler
from app.nlp_tasks.keyword_extractor import get_keyword_extractor_instance, KeywordExtractor
from app.db_connector.pg_connector import connect_db as connect_pg, close_db as close_pg, store_nlp_analysis_results
from app.services.scheduler_service import start_scheduler, stop_scheduler
# from app.main_processor import scheduled_nlp_job # Not directly used by main.py, but by scheduler_service

# --- IMPORT THE CENTRALIZED PIPELINE FUNCTION ---
from app.services.analysis_pipeline import execute_nlp_pipeline 

logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL.upper())

# Global variables to hold initialized NLP model instances
# These will be set during startup in the lifespan manager
sentiment_analyzer_global: Optional[SentimentAnalyzer] = None
topic_modeler_global: Optional[TopicModeler] = None
keyword_extractor_global: Optional[KeywordExtractor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{settings.SERVICE_NAME} starting up...")
    global sentiment_analyzer_global, topic_modeler_global, keyword_extractor_global

    # 1. Load NLP Models
    try:
        # Initialize and assign to global variables
        sentiment_analyzer_global = get_sentiment_analyzer_instance()
        topic_modeler_global = get_topic_modeler_instance() # This loads SBERT and BERTopic
        keyword_extractor_global = get_keyword_extractor_instance()

        # Perform readiness checks
        if not sentiment_analyzer_global.classifier:
            raise RuntimeError("Sentiment Analyzer model failed to load or classifier not initialized.")
        if not topic_modeler_global.topic_model: # Main BERTopic model
            raise RuntimeError(f"BERTopic model at {settings.BERTOPIC_MODEL_PATH} could not be loaded.")
        if not topic_modeler_global.embedding_model_instance: # SBERT model used by BERTopic
             raise RuntimeError(f"SBERT embedding model ('{settings.SBERT_MODEL_NAME}') for BERTopic failed to load.")
        if topic_modeler_global.topic_model and not topic_modeler_global.topic_model.embedding_model:
            # This check is if BERTopic loaded but somehow didn't get/retain the embedding model we passed
            logger.warning("BERTopic model loaded, but its internal SBERT embedding_model might be missing or mismatched. This could be problematic.")
            # Depending on strictness, you might raise RuntimeError here too.
            # For now, primary check is on topic_modeler_global.embedding_model_instance

        logger.info("NLP models initialized successfully.")
    except Exception as e:
        logger.critical(f"Critical error during NLP model initialization: {e}", exc_info=True)
        raise RuntimeError(f"NLP model initialization failed: {e}") from e # Halt startup

    # 2. Connect to Database
    try:
        await connect_pg() # Connect to PostgreSQL
    except Exception as e:
        logger.critical(f"Database connection failed during startup: {e}", exc_info=True)
        # Attempt to clean up if pool was partially created by connect_pg before raising
        try:
            await close_pg()
        except Exception as close_e:
            logger.error(f"Error during DB cleanup after connection failure: {close_e}")
        raise RuntimeError(f"Database connection failed: {e}") from e
    
    # 3. Start APScheduler
    try:
        await start_scheduler() # This will add and start the scheduled_nlp_job
    except Exception as e:
        logger.error(f"APScheduler failed to start: {e}", exc_info=True)
        # If scheduler is critical for the service's main operation, re-raise
        # For an API-first service that also has a background job, you might allow startup.
        # raise RuntimeError(f"APScheduler failed to start: {e}") from e 

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
    lifespan=lifespan 
)

@app.post("/analyze", response_model=NLPAnalysisResponse)
async def analyze_text_endpoint(
    request_data: NLPAnalysisRequest,
    # These Depends will now correctly use the globally initialized instances
    # The getter functions just return these global instances.
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer_instance),
    topic_modeler: TopicModeler = Depends(get_topic_modeler_instance),
    keyword_extractor: KeywordExtractor = Depends(get_keyword_extractor_instance)
):
    logger.info(f"API /analyze request for raw_mongo_id: {request_data.raw_mongo_id}")
    
    # Check if models are ready (they should be if startup succeeded)
    if not sentiment_analyzer.classifier or \
       not topic_modeler.topic_model or \
       not topic_modeler.embedding_model_instance: # Check the SBERT instance directly
        logger.error("One or more NLP models are not ready. API call cannot proceed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP models are not ready or failed to load during startup."
        )

    try:
        # --- CALL THE CENTRALIZED NLP PIPELINE ---
        nlp_response = await execute_nlp_pipeline(
            request_data=request_data,
            sentiment_analyzer=sentiment_analyzer,
            topic_modeler=topic_modeler,
            keyword_extractor=keyword_extractor
        )
        
        # Store the result from the API call as well
        if nlp_response: # Only store if pipeline returned a valid response
            try:
                await store_nlp_analysis_results([nlp_response])
                logger.info(f"API call result for {request_data.raw_mongo_id} stored/updated in DB.")
            except Exception as db_e:
                logger.error(f"Failed to store API call result for {request_data.raw_mongo_id} in DB: {db_e}", exc_info=True)
                # Log the error but still return the NLP response to the client,
                # as the analysis itself was successful.
                # The background job would eventually pick it up if it was from preprocessor,
                # but for direct API calls, this result might be lost if not stored.
                # Consider if this storage failure should make the API call fail (e.g., return 500).
                # For now, we prioritize returning the analysis result.
        else:
            # This case should ideally be handled within execute_nlp_pipeline by returning
            # an NLPAnalysisResponse with an errors field, or raising a specific exception.
            logger.error(f"NLP pipeline returned None for {request_data.raw_mongo_id}, indicating an internal error in the pipeline.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="NLP analysis pipeline failed to produce a result."
            )

        return nlp_response

    except HTTPException as http_exc: # Re-raise HTTPExceptions from pipeline if any
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during API analysis for {request_data.raw_mongo_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during NLP analysis.")


if __name__ == "__main__":
    import uvicorn
    # The SBERT_MODEL_NAME and other model names are now part of the `settings` object
    # and are used during the initialization of SentimentAnalyzer and TopicModeler.
    # The TopicModeler itself will log if settings.SBERT_MODEL_NAME is missing or loading fails.
    logger.info(f"Starting {settings.SERVICE_NAME} for local development...")
    # Use a dedicated SERVICE_PORT setting for clarity if you add it to config.py and .env
    # Defaulting to 8001 to match your Dockerfile EXPOSE and CMD.
    service_port = 8001 
    try:
        # If you add SERVICE_PORT to settings:
        # service_port = settings.SERVICE_PORT
        pass
    except AttributeError:
        logger.warning(f"SERVICE_PORT not in settings, defaulting to {service_port} for local run.")

    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=service_port, 
        log_level=settings.LOG_LEVEL.lower(), 
        reload=True # Reload is fine for local development
    )