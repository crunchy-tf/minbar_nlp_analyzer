# nlp_analyzer_service/app/main_processor.py
import asyncio
import json # <<< Ensure json is imported
from loguru import logger
from typing import List, Dict, Any, Optional

from app.config import settings # Use Pydantic settings
from app.db_connector.pg_connector import (
    fetch_preprocessed_docs_for_nlp,
    mark_docs_as_nlp_processed_in_source,
    store_nlp_analysis_results
)
from app.models import NLPAnalysisRequest, NLPAnalysisResponse # Import Pydantic models
# Import your actual analysis tool instances via their getter functions
from app.nlp_tasks.sentiment_analyzer import get_sentiment_analyzer_instance, SentimentAnalyzer
from app.nlp_tasks.topic_modeler import get_topic_modeler_instance, TopicModeler
from app.nlp_tasks.keyword_extractor import get_keyword_extractor_instance, KeywordExtractor
# Import the centralized NLP pipeline execution function
from app.services.analysis_pipeline import execute_nlp_pipeline

async def _process_document_with_pipeline(
    doc_data: Dict[str, Any], # Data fetched from the preprocessor's output DB
    sentiment_analyzer: SentimentAnalyzer,
    topic_modeler: TopicModeler,
    keyword_extractor: KeywordExtractor
) -> Optional[NLPAnalysisResponse]:
    """
    Prepares the NLPAnalysisRequest from DB data and calls the centralized execute_nlp_pipeline.
    """
    raw_mongo_id_for_logging = doc_data.get('raw_mongo_id', 'UnknownRawMongoID')
    try:
        # Parse tokens_processed and lemmas from JSON string to list
        tokens_processed_list: Optional[List[str]] = None
        tokens_processed_str = doc_data.get('tokens_processed')
        if isinstance(tokens_processed_str, str):
            try:
                tokens_processed_list = json.loads(tokens_processed_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode tokens_processed JSON string for doc_id {raw_mongo_id_for_logging}: '{tokens_processed_str}'")
        elif isinstance(tokens_processed_str, list): # Should ideally not happen if PG returns JSONB as string to asyncpg
             tokens_processed_list = tokens_processed_str
        elif tokens_processed_str is not None: # It's some other type, log a warning
            logger.warning(f"tokens_processed for doc_id {raw_mongo_id_for_logging} is of unexpected type {type(tokens_processed_str)}. Expected str or list.")


        lemmas_list: Optional[List[str]] = None
        lemmas_str = doc_data.get('lemmas')
        if isinstance(lemmas_str, str):
            try:
                lemmas_list = json.loads(lemmas_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode lemmas JSON string for doc_id {raw_mongo_id_for_logging}: '{lemmas_str}'")
        elif isinstance(lemmas_str, list): # Should ideally not happen
             lemmas_list = lemmas_str
        elif lemmas_str is not None:
            logger.warning(f"lemmas for doc_id {raw_mongo_id_for_logging} is of unexpected type {type(lemmas_str)}. Expected str or list.")


        # Map keys from preprocessor's table (doc_data) to NLPAnalysisRequest model fields
        request_input = NLPAnalysisRequest(
            raw_mongo_id=str(doc_data.get('raw_mongo_id')), # Ensure string
            source=doc_data.get('source', 'unknown_source_type'),
            original_timestamp=doc_data.get('original_timestamp'), # Should be datetime from DB
            retrieved_by_keyword=doc_data.get('retrieved_by_keyword', 'unknown_keyword'),
            keyword_language=doc_data.get('keyword_language', 'un'), # Provide default if might be missing
            keyword_concept_id=doc_data.get('keyword_concept_id'), 
            detected_language=doc_data.get('detected_language'),
            cleaned_text=doc_data.get('cleaned_text', ''), # Ensure not None
            tokens_processed=tokens_processed_list, # <<< USE PARSED LIST
            lemmas=lemmas_list,                   # <<< USE PARSED LIST
            original_url=doc_data.get('original_url')
        )

        # Call the centralized pipeline function
        nlp_response = await execute_nlp_pipeline(
            request_data=request_input,
            sentiment_analyzer=sentiment_analyzer,
            topic_modeler=topic_modeler,
            keyword_extractor=keyword_extractor
        )
        return nlp_response

    except Exception as e:
        # This catches errors during NLPAnalysisRequest creation or if execute_nlp_pipeline itself raises an unhandled error
        logger.error(f"Error preparing request or critical error in pipeline for doc_id {raw_mongo_id_for_logging}: {e}", exc_info=True)
        return None


async def scheduled_nlp_job():
    logger.info("--- Starting scheduled NLP analysis job ---")

    sentiment_analyzer = get_sentiment_analyzer_instance()
    topic_modeler = get_topic_modeler_instance()
    keyword_extractor = get_keyword_extractor_instance()

    # Readiness checks (crucial before processing)
    if not sentiment_analyzer or not sentiment_analyzer.classifier: # Added check for sentiment_analyzer itself
        logger.error("Sentiment Analyzer model not ready or not initialized. Aborting NLP job.")
        return
    if not topic_modeler or not topic_modeler.topic_model or not topic_modeler.embedding_model_instance :
        logger.error("Topic Modeler (BERTopic or SBERT embedding) not ready or not initialized. Aborting NLP job.")
        return
    if not keyword_extractor: # Added check for keyword_extractor itself
        logger.error("Keyword Extractor not initialized. Aborting NLP job.")
        return


    docs_to_process_dicts = await fetch_preprocessed_docs_for_nlp(settings.NLP_BATCH_SIZE)

    if not docs_to_process_dicts:
        logger.info("No new documents from preprocessor to analyze in this cycle.")
        logger.info("--- NLP analysis job finished (no data) ---")
        return

    logger.info(f"Fetched {len(docs_to_process_dicts)} documents for NLP analysis.")

    analysis_tasks = [
        _process_document_with_pipeline(
            doc_data,
            sentiment_analyzer,
            topic_modeler,
            keyword_extractor
        ) for doc_data in docs_to_process_dicts
    ]

    results_or_exceptions: List[Optional[NLPAnalysisResponse]] = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    successful_nlp_responses: List[NLPAnalysisResponse] = []
    source_doc_pg_ids_processed_ok: List[int] = []
    source_doc_pg_ids_failed_nlp: List[int] = [] # For documents where NLPAnalysisRequest creation or pipeline failed

    for i, res_or_exc in enumerate(results_or_exceptions):
        source_doc_pg_id = docs_to_process_dicts[i].get('id') # This is the PK from processed_documents
        raw_mongo_id_for_logging = docs_to_process_dicts[i].get('raw_mongo_id', 'UnknownRawMongoID')

        if source_doc_pg_id is None: # Should not happen if query selects id
            logger.error(f"Source document (raw_mongo_id: {raw_mongo_id_for_logging}) at index {i} missing 'id' from DB fetch. Cannot track for status update.")
            continue

        if isinstance(res_or_exc, Exception):
            # This exception would be from asyncio.gather if _process_document_with_pipeline had an unhandled one,
            # but _process_document_with_pipeline now catches its own errors and returns None.
            # So, this branch might be less likely to be hit for pipeline errors, more for gather issues.
            logger.error(f"Unhandled exception during NLP pipeline task for source_doc_id {source_doc_pg_id} (raw_mongo_id: {raw_mongo_id_for_logging}): {res_or_exc}", exc_info=True)
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)
        elif isinstance(res_or_exc, NLPAnalysisResponse):
            successful_nlp_responses.append(res_or_exc)
            source_doc_pg_ids_processed_ok.append(source_doc_pg_id) # Track by source DB ID
        else: # res_or_exc is None because _process_document_with_pipeline returned None
            logger.warning(f"NLP analysis pipeline (or request prep) returned None for source_doc_id {source_doc_pg_id} (raw_mongo_id: {raw_mongo_id_for_logging}). Marking as failed_nlp_analysis.")
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)

    if successful_nlp_responses:
        stored_count = await store_nlp_analysis_results(successful_nlp_responses)
        logger.info(f"Attempted to store {len(successful_nlp_responses)} NLP results, {stored_count} operations reported by DB.")

        # Logic for marking source documents based on storage success
        # For simplicity, if any storage was attempted, we'll mark based on the original successful_nlp_responses list.
        # A more complex logic might involve checking which specific inserts succeeded if store_nlp_analysis_results could partially fail.
        if stored_count > 0 and source_doc_pg_ids_processed_ok:
             if stored_count == len(source_doc_pg_ids_processed_ok):
                 await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='completed')
             else:
                 logger.warning(f"Partial success storing NLP results ({stored_count}/{len(source_doc_pg_ids_processed_ok)}). Marking source docs as 'pending_nlp_store_verification'.")
                 await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='pending_nlp_store_verification')
        elif source_doc_pg_ids_processed_ok: # NLP was ok, but 0 stored
             logger.warning(f"NLP processing succeeded for {len(source_doc_pg_ids_processed_ok)} docs, but 0 results were stored. Marking source docs as 'pending_nlp_store_verification'.")
             await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='pending_nlp_store_verification')


    if source_doc_pg_ids_failed_nlp:
        await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_failed_nlp, status='failed_nlp_analysis')

    logger.info(f"--- NLP analysis job finished. Documents evaluated: {len(docs_to_process_dicts)}, Succeeded NLP processing (and attempted store): {len(successful_nlp_responses)}, Failed before/during NLP pipeline: {len(source_doc_pg_ids_failed_nlp)} ---")