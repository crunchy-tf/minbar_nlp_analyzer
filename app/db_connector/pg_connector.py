# nlp_analyzer_service/app/db_connector/pg_connector.py
import asyncpg
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime, timezone # Import timezone
import json # For storing JSONB data

from app.config import settings # Use the new Pydantic settings
from app.models import NLPAnalysisResponse

_pool: Optional[asyncpg.Pool] = None
_source_table_checked_for_status_field = False
_target_table_created_and_checked = False


async def connect_db():
    global _pool, _source_table_checked_for_status_field, _target_table_created_and_checked
    if _pool and not getattr(_pool, '_closed', True):
        logger.debug("PostgreSQL connection pool already established.")
        if not _source_table_checked_for_status_field: await ensure_status_field_in_source_table()
        if not _target_table_created_and_checked: await create_target_nlp_results_table_if_not_exists()
        return

    logger.info(f"Connecting to PostgreSQL database via DSN derived from target settings: {settings.postgres_dsn_asyncpg}")
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn_asyncpg, # This DSN points to TARGET_POSTGRES_DB
            min_size=2,
            max_size=10
        )
        # Both source and target tables are in the same DB for Option 2
        await ensure_status_field_in_source_table()
        await create_target_nlp_results_table_if_not_exists()
        logger.success("PostgreSQL connection pool established and tables checked/prepared.")
    except Exception as e:
        logger.critical(f"Failed to connect to PostgreSQL or prepare tables: {e}", exc_info=True)
        _pool = None
        _source_table_checked_for_status_field = False
        _target_table_created_and_checked = False
        raise ConnectionError(f"Could not connect to PostgreSQL or prepare tables: {e}") from e

async def close_db():
    global _pool, _source_table_checked_for_status_field, _target_table_created_and_checked
    if _pool:
        logger.info("Closing PostgreSQL connection pool.")
        await _pool.close()
        _pool = None
        _source_table_checked_for_status_field = False
        _target_table_created_and_checked = False
        logger.success("PostgreSQL connection pool closed.")

async def get_pool() -> asyncpg.Pool:
    if _pool is None or getattr(_pool, '_closed', True):
        logger.warning("PostgreSQL pool is None or closed. Attempting to (re)initialize...")
        await connect_db()
    if _pool is None: # Check again after attempt
        raise ConnectionError("PostgreSQL pool unavailable after re-initialization attempt.")
    return _pool

async def ensure_status_field_in_source_table():
    global _source_table_checked_for_status_field
    if _source_table_checked_for_status_field: return

    pool = await get_pool()
    field_name = settings.NLP_ANALYZER_STATUS_FIELD_IN_SOURCE
    table_name = settings.SOURCE_POSTGRES_TABLE # This is 'processed_documents'
    # This check/alter happens on the DB specified by settings.postgres_dsn_asyncpg
    # which should be minbar_processed_data
    logger.info(f"Ensuring status field '{field_name}' exists in source table '{table_name}' in DB '{settings.SOURCE_POSTGRES_DB}'...")
    try:
        async with pool.acquire() as conn:
            exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = $1 AND column_name = $2
                );
            """, table_name, field_name)
            if not exists:
                logger.info(f"Field '{field_name}' not found in '{table_name}'. Attempting to add it.")
                await conn.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS {field_name} VARCHAR(20) DEFAULT 'pending',
                    ADD COLUMN IF NOT EXISTS {field_name}_timestamp TIMESTAMPTZ;
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{field_name.replace("_","-")}
                    ON {table_name} ({field_name});
                """) # Ensure index name is valid
                logger.success(f"Field '{field_name}' and timestamp added to '{table_name}'.")
            else:
                logger.debug(f"Field '{field_name}' already exists in '{table_name}'.")
        _source_table_checked_for_status_field = True
    except Exception as e:
        logger.error(f"Error ensuring status field in source table '{table_name}': {e}", exc_info=True)
        _source_table_checked_for_status_field = False
        raise

async def create_target_nlp_results_table_if_not_exists():
    global _target_table_created_and_checked
    if _target_table_created_and_checked: return

    pool = await get_pool()
    table_name = settings.TARGET_POSTGRES_TABLE # This is 'document_nlp_outputs'
    # This table is created in the DB specified by settings.postgres_dsn_asyncpg
    # which should be minbar_processed_data
    logger.info(f"Checking/Creating target NLP results table '{table_name}' in DB '{settings.TARGET_POSTGRES_DB}'...")
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        -- processed_document_pg_id INTEGER REFERENCES {settings.SOURCE_POSTGRES_TABLE}(id) ON DELETE CASCADE, -- Optional FK
        raw_mongo_id VARCHAR(24) NOT NULL UNIQUE, -- From NLPAnalysisResponse
        source TEXT,
        original_timestamp TIMESTAMPTZ,
        retrieved_by_keyword TEXT,
        processing_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        detected_language CHAR(2),
        overall_sentiment JSONB,
        assigned_topics JSONB,
        extracted_keywords_frequency JSONB,
        sentiment_on_extracted_keywords_summary JSONB,
        analysis_errors TEXT,
        nlp_model_version VARCHAR(100),
        signal_extraction_status VARCHAR(20) DEFAULT 'pending',
        signal_extraction_timestamp TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_nlp_doc_out_raw_mongo_id ON {table_name}(raw_mongo_id);
    CREATE INDEX IF NOT EXISTS idx_nlp_doc_out_orig_ts ON {table_name}(original_timestamp);
    CREATE INDEX IF NOT EXISTS idx_nlp_doc_out_sig_ext_status ON {table_name}(signal_extraction_status, original_timestamp);
    CREATE INDEX IF NOT EXISTS idx_nlp_doc_out_topics ON {table_name} USING GIN (assigned_topics);
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(create_table_sql)
        logger.success(f"Target NLP results table '{table_name}' checked/created successfully.")
        _target_table_created_and_checked = True
    except Exception as e:
        logger.error(f"Error creating/checking target NLP results table '{table_name}': {e}", exc_info=True)
        _target_table_created_and_checked = False
        raise

async def fetch_preprocessed_docs_for_nlp(limit: int) -> List[Dict[str, Any]]:
    pool = await get_pool()
    # This queries the SOURCE table (processed_documents)
    query = f"""
        SELECT id, raw_mongo_id, source, original_timestamp, retrieved_by_keyword, keyword_language,
               detected_language, cleaned_text, tokens_processed, lemmas, original_url
        FROM {settings.SOURCE_POSTGRES_TABLE}
        WHERE {settings.NLP_ANALYZER_STATUS_FIELD_IN_SOURCE} IS DISTINCT FROM 'completed' AND
              {settings.NLP_ANALYZER_STATUS_FIELD_IN_SOURCE} IS DISTINCT FROM 'failed_permanent' 
        ORDER BY original_timestamp ASC
        LIMIT $1;
    """
    try:
        records = await pool.fetch(query, limit)
        logger.info(f"Fetched {len(records)} preprocessed documents for NLP analysis from '{settings.SOURCE_POSTGRES_TABLE}'.")
        return [dict(record) for record in records]
    except Exception as e:
        logger.error(f"Error fetching preprocessed documents from '{settings.SOURCE_POSTGRES_TABLE}': {e}", exc_info=True)
        return []

async def mark_docs_as_nlp_processed_in_source(
    doc_pg_ids: List[int], # Use primary key 'id' from source_postgres_table
    status: str = 'completed'
) -> int:
    if not settings.MARK_AS_NLP_PROCESSED_IN_SOURCE_DB or not doc_pg_ids:
        if not doc_pg_ids: logger.debug("No source document IDs to mark for NLP status.")
        return 0
        
    pool = await get_pool()
    update_query = f"""
        UPDATE {settings.SOURCE_POSTGRES_TABLE}
        SET 
            {settings.NLP_ANALYZER_STATUS_FIELD_IN_SOURCE} = $1,
            {settings.NLP_ANALYZER_STATUS_FIELD_IN_SOURCE}_timestamp = $2
        WHERE id = ANY($3::int[]);
    """
    try:
        # Ensure doc_pg_ids are actually integers
        valid_doc_pg_ids = [int(id_val) for id_val in doc_pg_ids if id_val is not None]
        if not valid_doc_pg_ids:
            logger.warning("No valid integer document IDs provided for marking NLP status.")
            return 0

        result_status_str = await pool.execute(update_query, status, datetime.now(timezone.utc), valid_doc_pg_ids)
        # Format of result_status_str is typically "UPDATE N"
        updated_count = 0
        if result_status_str and result_status_str.startswith("UPDATE "):
            try:
                updated_count = int(result_status_str.split(" ")[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not parse update count from status: {result_status_str}")
                # Assume all were attempted if no error, or rely on actual modified count if available from driver
                updated_count = len(valid_doc_pg_ids) # Fallback assumption

        logger.info(f"Marked {updated_count} documents as NLP '{status}' in source table '{settings.SOURCE_POSTGRES_TABLE}'.")
        return updated_count
    except Exception as e:
        logger.error(f"Error marking documents as NLP processed in source table '{settings.SOURCE_POSTGRES_TABLE}': {e}", exc_info=True)
        return 0

async def store_nlp_analysis_results(results: List[NLPAnalysisResponse]) -> int:
    if not results:
        return 0
    pool = await get_pool()
    
    data_to_insert = []
    for res in results:
        # Ensure all JSONB fields are properly json.dumps'd
        data_to_insert.append((
            res.raw_mongo_id,
            res.source,
            res.original_timestamp,
            res.retrieved_by_keyword,
            res.processing_timestamp,
            res.detected_language,
            json.dumps([s.model_dump() for s in res.overall_sentiment]) if res.overall_sentiment else None,
            json.dumps([t.model_dump(exclude_none=True) for t in res.assigned_topics]) if res.assigned_topics else None, # Exclude None probabilities
            json.dumps([kf.model_dump() for kf in res.extracted_keywords_frequency]) if res.extracted_keywords_frequency else None,
            json.dumps([s.model_dump() for s in res.sentiment_on_extracted_keywords_summary]) if res.sentiment_on_extracted_keywords_summary else None,
            json.dumps(res.analysis_errors) if res.analysis_errors else None,
            "BERTopic_SBERT_XLM-R_v1" # Example model version string
        ))

    insert_query = f"""
        INSERT INTO {settings.TARGET_POSTGRES_TABLE} (
            raw_mongo_id, source, original_timestamp, retrieved_by_keyword, processing_timestamp,
            detected_language, overall_sentiment, assigned_topics, extracted_keywords_frequency,
            sentiment_on_extracted_keywords_summary, analysis_errors, nlp_model_version
            -- signal_extraction_status is defaulted to 'pending' by table DDL
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        )
        ON CONFLICT (raw_mongo_id) DO UPDATE SET 
            source = EXCLUDED.source,
            original_timestamp = EXCLUDED.original_timestamp,
            retrieved_by_keyword = EXCLUDED.retrieved_by_keyword,
            processing_timestamp = EXCLUDED.processing_timestamp,
            detected_language = EXCLUDED.detected_language,
            overall_sentiment = EXCLUDED.overall_sentiment,
            assigned_topics = EXCLUDED.assigned_topics,
            extracted_keywords_frequency = EXCLUDED.extracted_keywords_frequency,
            sentiment_on_extracted_keywords_summary = EXCLUDED.sentiment_on_extracted_keywords_summary,
            analysis_errors = EXCLUDED.analysis_errors,
            nlp_model_version = EXCLUDED.nlp_model_version,
            signal_extraction_status = 'pending', -- Reset for Signal Extraction on re-analysis
            signal_extraction_timestamp = NULL;
    """
    try:
        async with pool.acquire() as conn:
            status_command = await conn.executemany(insert_query, data_to_insert)
            inserted_count = len(data_to_insert) # Assume all were attempted
            logger.info(f"Stored/Updated {inserted_count} NLP analysis results in '{settings.TARGET_POSTGRES_TABLE}'. Status: {status_command}")
            return inserted_count
    except Exception as e:
        logger.error(f"Error storing NLP analysis results: {e}", exc_info=True)
        return 0