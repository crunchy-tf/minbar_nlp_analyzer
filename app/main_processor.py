# nlp_analyzer_service/app/main_processor.py
import asyncio
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
    try:
        # Map keys from preprocessor's table (doc_data) to NLPAnalysisRequest model fields
        request_input = NLPAnalysisRequest(
            raw_mongo_id=str(doc_data.get('raw_mongo_id')), # Ensure string
            source=doc_data.get('source', 'unknown_source_type'),
            original_timestamp=doc_data.get('original_timestamp'), # Should be datetime from DB
            retrieved_by_keyword=doc_data.get('retrieved_by_keyword', 'unknown_keyword'),
            keyword_language=doc_data.get('keyword_language', 'un'), # Provide default if might be missing
            keyword_concept_id=doc_data.get('keyword_concept_id'), # <<< UPDATED (ensure this is selected by fetch_preprocessed_docs_for_nlp)
            detected_language=doc_data.get('detected_language'),
            cleaned_text=doc_data.get('cleaned_text', ''), # Ensure not None
            tokens_processed=doc_data.get('tokens_processed'), # List or None
            lemmas=doc_data.get('lemmas'), # List or None
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
        logger.error(f"Error preparing request or critical error in pipeline for doc_id {doc_data.get('raw_mongo_id')}: {e}", exc_info=True)
        return None


async def scheduled_nlp_job():
    logger.info("--- Starting scheduled NLP analysis job ---")

    sentiment_analyzer = get_sentiment_analyzer_instance()
    topic_modeler = get_topic_modeler_instance()
    keyword_extractor = get_keyword_extractor_instance()

    if not sentiment_analyzer.classifier:
        logger.error("Sentiment Analyzer model not ready. Aborting NLP job.")
        return
    if not topic_modeler.topic_model or not topic_modeler.embedding_model_instance : # Check SBERT instance directly
        logger.error("Topic Modeler (BERTopic or SBERT embedding) not ready. Aborting NLP job.")
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
    source_doc_pg_ids_failed_nlp: List[int] = []

    for i, res_or_exc in enumerate(results_or_exceptions):
        source_doc_pg_id = docs_to_process_dicts[i].get('id')
        raw_mongo_id_for_logging = docs_to_process_dicts[i].get('raw_mongo_id', 'UnknownRawMongoID')

        if source_doc_pg_id is None:
            logger.error(f"Source document (raw_mongo_id: {raw_mongo_id_for_logging}) at index {i} missing 'id' from DB fetch. Cannot track for status update.")
            continue

        if isinstance(res_or_exc, Exception):
            logger.error(f"Unhandled exception during NLP pipeline for source_doc_id {source_doc_pg_id} (raw_mongo_id: {raw_mongo_id_for_logging}): {res_or_exc}", exc_info=True)
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)
        elif isinstance(res_or_exc, NLPAnalysisResponse):
            successful_nlp_responses.append(res_or_exc)
            source_doc_pg_ids_processed_ok.append(source_doc_pg_id)
        else:
            logger.warning(f"NLP analysis pipeline returned None (internal error or failed request prep) for source_doc_id {source_doc_pg_id} (raw_mongo_id: {raw_mongo_id_for_logging}).")
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)

    if successful_nlp_responses:
        stored_count = await store_nlp_analysis_results(successful_nlp_responses)
        logger.info(f"Attempted to store {len(successful_nlp_responses)} NLP results, {stored_count} operations reported by DB.")

        if stored_count == len(successful_nlp_responses) and source_doc_pg_ids_processed_ok:
            await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='completed')
        elif source_doc_pg_ids_processed_ok:
            logger.warning(f"NLP processing succeeded for {len(source_doc_pg_ids_processed_ok)} docs, but storage count ({stored_count}) mismatched or was zero. Marking source docs as 'pending_nlp_store_verification'.")
            await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='pending_nlp_store_verification')

    if source_doc_pg_ids_failed_nlp:
        await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_failed_nlp, status='failed_nlp_analysis')

    logger.info(f"--- NLP analysis job finished. Documents evaluated: {len(docs_to_process_dicts)}, Succeeded NLP processing: {len(successful_nlp_responses)}, Failed NLP processing: {len(source_doc_pg_ids_failed_nlp)} ---")