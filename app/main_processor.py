# nlp_analyzer_service/app/main_processor.py
import asyncio
from loguru import logger
from typing import List, Dict, Any, Optional

from app.config import settings
from app.db_connector.pg_connector import (
    fetch_preprocessed_docs_for_nlp,
    mark_docs_as_nlp_processed_in_source,
    store_nlp_analysis_results
)
from app.models import NLPAnalysisRequest, NLPAnalysisResponse # Import Pydantic models
# Import your actual analysis functions/classes as singletons or via getters
from app.nlp_tasks.sentiment_analyzer import get_sentiment_analyzer_instance, SentimentAnalyzer
from app.nlp_tasks.topic_modeler import get_topic_modeler_instance, TopicModeler
from app.nlp_tasks.keyword_extractor import get_keyword_extractor_instance, KeywordExtractor
# We need a way to call the core logic of the /analyze endpoint
# Option 1: Import and call the endpoint function directly (if it doesn't rely too much on FastAPI specifics like Request objects not passed)
# from app.main import analyze_text 
# Option 2: Refactor the core logic of analyze_text into a separate callable function

async def _perform_single_doc_analysis(
    doc_data: Dict[str, Any],
    sentiment_analyzer: SentimentAnalyzer, 
    topic_modeler: TopicModeler, 
    keyword_extractor: KeywordExtractor
) -> Optional[NLPAnalysisResponse]:
    """
    Prepares request and calls the core analysis logic for a single document.
    This function encapsulates the logic originally in the /analyze endpoint.
    """
    try:
        # Map keys from preprocessor's table to NLPAnalysisRequest model
        request_input = NLPAnalysisRequest(
            raw_mongo_id=str(doc_data.get('raw_mongo_id')), # Ensure string
            source=doc_data.get('source', 'unknown_source_type'),
            original_timestamp=doc_data.get('original_timestamp'),
            retrieved_by_keyword=doc_data.get('retrieved_by_keyword', 'unknown'),
            keyword_language=doc_data.get('keyword_language', 'un'),
            detected_language=doc_data.get('detected_language'),
            cleaned_text=doc_data.get('cleaned_text', ''),
            tokens_processed=doc_data.get('tokens_processed'),
            lemmas=doc_data.get('lemmas'),
            original_url=doc_data.get('original_url')
        )

        # --- Replicate logic from app.main.analyze_text ---
        errors = []
        overall_sentiment_scores_raw = sentiment_analyzer.analyze(request_input.cleaned_text)
        overall_sentiment_scores = [NLPAnalysisResponse.model_fields['overall_sentiment'].annotation.__args__[0](**s) for s in overall_sentiment_scores_raw] if overall_sentiment_scores_raw else []
        if not overall_sentiment_scores_raw and request_input.cleaned_text.strip():
            errors.append("Overall sentiment analysis failed or returned no result.")

        assigned_doc_topics = []
        if topic_modeler.topic_model and topic_modeler.topic_model.embedding_model:
            topic_result = topic_modeler.get_topics_for_doc(request_input.cleaned_text)
            if topic_result:
                pred_topic_ids, pred_topic_probs_distributions = topic_result
                for i, topic_id in enumerate(pred_topic_ids):
                    # ... (copy the topic processing logic from your main.py analyze_text endpoint here) ...
                    # Make sure to use NLPAnalysisResponse.model_fields['assigned_topics'].annotation.__args__[0]
                    # for TopicInfo model instantiation if needed.
                    # Simplified for brevity, ensure full logic is copied and adapted:
                    topic_name_str = topic_modeler.get_topic_name(topic_id)
                    topic_keywords = topic_modeler.get_topic_details(topic_id) or []
                    prob = None
                    if pred_topic_probs_distributions and i < len(pred_topic_probs_distributions) and pred_topic_probs_distributions[i] is not None:
                        prob_data = pred_topic_probs_distributions[i]
                        if isinstance(prob_data, (float, int)): prob = float(prob_data)
                        elif hasattr(prob_data, 'max'): prob = float(prob_data.max())


                    assigned_doc_topics.append(
                         NLPAnalysisResponse.model_fields['assigned_topics'].annotation.__args__[0]( # TopicInfo
                            id=int(topic_id), 
                            name=topic_name_str, 
                            keywords=topic_keywords,
                            probability=prob
                        )
                    )
            elif request_input.cleaned_text.strip():
                errors.append("Topic modeling returned no result.")
        else:
            errors.append("Topic modeler not ready.")

        input_tokens_for_freq = request_input.lemmas if request_input.lemmas else request_input.tokens_processed
        freq_keywords_raw = keyword_extractor.extract_from_tokens_frequency(input_tokens_for_freq)
        extracted_keywords_freq = [NLPAnalysisResponse.model_fields['extracted_keywords_frequency'].annotation.__args__[0](keyword=kw, frequency=f) for kw, f in freq_keywords_raw]

        sentiment_on_keywords_summary = None
        if extracted_keywords_freq:
            top_overall_keywords_text = " ".join([kf.keyword for kf in extracted_keywords_freq[:keyword_extractor.top_n]])
            if top_overall_keywords_text.strip():
                kw_summary_sentiment_raw = sentiment_analyzer.analyze(top_overall_keywords_text)
                sentiment_on_keywords_summary = [NLPAnalysisResponse.model_fields['sentiment_on_extracted_keywords_summary'].annotation.__args__[0].__args__[0](**s) for s in kw_summary_sentiment_raw] if kw_summary_sentiment_raw else []
        
        return NLPAnalysisResponse(
            raw_mongo_id=request_input.raw_mongo_id,
            source=request_input.source,
            original_timestamp=request_input.original_timestamp,
            retrieved_by_keyword=request_input.retrieved_by_keyword,
            detected_language=request_input.detected_language,
            overall_sentiment=overall_sentiment_scores,
            assigned_topics=assigned_doc_topics,
            extracted_keywords_frequency=extracted_keywords_freq,
            sentiment_on_extracted_keywords_summary=sentiment_on_keywords_summary,
            analysis_errors=errors if errors else None
        )
        # --- End replicated logic ---

    except Exception as e:
        logger.error(f"Error during NLP analysis for doc_id {doc_data.get('raw_mongo_id')}: {e}", exc_info=True)
        return None

async def scheduled_nlp_job():
    logger.info("--- Starting scheduled NLP analysis job ---")
    
    sa = get_sentiment_analyzer_instance()
    tm = get_topic_modeler_instance()
    ke = get_keyword_extractor_instance()

    if not sa.classifier or not tm.topic_model or not tm.topic_model.embedding_model:
        logger.error("One or more NLP models are not ready. Aborting NLP job.")
        return

    docs_to_process_dicts = await fetch_preprocessed_docs_for_nlp(settings.NLP_BATCH_SIZE)
    if not docs_to_process_dicts:
        logger.info("No new documents from preprocessor to analyze.")
        logger.info("--- NLP analysis job finished (no data) ---")
        return

    logger.info(f"Fetched {len(docs_to_process_dicts)} documents for NLP analysis.")
    
    analysis_tasks = [
        _perform_single_doc_analysis(doc_data, sa, tm, ke) for doc_data in docs_to_process_dicts
    ]
    
    results_or_exceptions: List[Optional[NLPAnalysisResponse]] = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    successful_nlp_responses: List[NLPAnalysisResponse] = []
    source_doc_pg_ids_processed_ok: List[int] = []
    source_doc_pg_ids_failed_nlp: List[int] = []

    for i, res_or_exc in enumerate(results_or_exceptions):
        # Assuming 'id' is the primary key of the source 'processed_documents' table
        source_doc_pg_id = docs_to_process_dicts[i].get('id') 
        if source_doc_pg_id is None:
            logger.error(f"Source document at index {i} (raw_mongo_id: {docs_to_process_dicts[i].get('raw_mongo_id')}) missing 'id' from DB fetch. Cannot track for status update.")
            continue # Skip this one if we can't get its source PK

        if isinstance(res_or_exc, Exception):
            logger.error(f"Unhandled exception during NLP analysis for source_doc_id {source_doc_pg_id}: {res_or_exc}", exc_info=True)
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)
        elif isinstance(res_or_exc, NLPAnalysisResponse):
            successful_nlp_responses.append(res_or_exc)
            source_doc_pg_ids_processed_ok.append(source_doc_pg_id)
        else: # res_or_exc is None, meaning _perform_single_doc_analysis had an internal error
            logger.warning(f"NLP analysis returned None (internal error) for source_doc_id {source_doc_pg_id}.")
            source_doc_pg_ids_failed_nlp.append(source_doc_pg_id)

    if successful_nlp_responses:
        stored_count = await store_nlp_analysis_results(successful_nlp_responses)
        logger.info(f"Attempted to store {len(successful_nlp_responses)} NLP results, {stored_count} succeeded.")
        if stored_count == len(successful_nlp_responses) and source_doc_pg_ids_processed_ok:
            # All successfully processed & stored results can be marked in source
            await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='completed')
        else:
            # Partial success or storage failure. Only mark those whose corresponding NLP result was part of the 'stored_count'.
            # This is complex. For now, if not all stored, don't mark any as 'completed' to ensure retry.
            # Or, more robustly, store_nlp_analysis_results could return IDs of successfully stored items.
            logger.warning("Not all NLP results may have been stored, or mismatch in counts. Source documents for successfully stored items will not be marked 'completed' to ensure data integrity / allow reprocessing of the batch.")
            # Potentially mark successfully processed and stored items as 'completed'
            # and others as 'failed_to_store_nlp_result'
            if source_doc_pg_ids_processed_ok: # The IDs that _were_ successfully NLP processed
                 await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_processed_ok, status='pending_store_verification')


    if source_doc_pg_ids_failed_nlp:
        await mark_docs_as_nlp_processed_in_source(source_doc_pg_ids_failed_nlp, status='failed_nlp_analysis')

    logger.info(f"--- NLP analysis job finished. Fetched: {len(docs_to_process_dicts)}, Succeeded NLP: {len(successful_nlp_responses)}, Failed NLP: {len(source_doc_pg_ids_failed_nlp)} ---")