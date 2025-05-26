# nlp_analyzer_service/app/config.py
import os
from typing import List, Union, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
import json
from loguru import logger

class Settings(BaseSettings):
    SERVICE_NAME: str = "Minbar NLP Analyzer Service"
    LOG_LEVEL: str = "INFO"

    # --- Model Paths & Names ---
    BERTOPIC_MODEL_FILENAME: str = Field(default="bertopic_model_final_guided_multilang_gensim.joblib")
    SBERT_MODEL_NAME: str = Field(default="paraphrase-multilingual-mpnet-base-v2")
    SENTIMENT_MODEL_NAME: str = Field(default="joeddav/xlm-roberta-large-xnli")
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Give BERTOPIC_MODEL_PATH a default string value. It will be overridden by the model_validator.
    # This satisfies Pydantic's initial validation that a 'str' field has a string-like value.
    BERTOPIC_MODEL_PATH: str = Field(default="dummy_path_will_be_overridden") 

    HEALTHCARE_SENTIMENT_LABELS: List[str] = [
        "Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"
    ]
    TOP_N_KEYWORDS: int = 10

    # --- Source DB (Data Preprocessor's Output in minbar_processed_data) ---
    SOURCE_POSTGRES_USER: str = Field(validation_alias='SOURCE_POSTGRES_USER')
    SOURCE_POSTGRES_PASSWORD: str = Field(validation_alias='SOURCE_POSTGRES_PASSWORD')
    SOURCE_POSTGRES_HOST: str = Field(validation_alias='SOURCE_POSTGRES_HOST')
    SOURCE_POSTGRES_PORT: int = Field(default=5432, validation_alias='SOURCE_POSTGRES_PORT')
    SOURCE_POSTGRES_DB: str = Field(validation_alias='SOURCE_POSTGRES_DB')
    SOURCE_POSTGRES_TABLE: str = Field(default="processed_documents", validation_alias='SOURCE_POSTGRES_TABLE')
    NLP_ANALYZER_STATUS_FIELD_IN_SOURCE: str = Field(default="nlp_analyzer_v1_status", validation_alias='NLP_ANALYZER_STATUS_FIELD_IN_SOURCE')

    # --- Target DB (This service's output table, in the SAME DB as source for Option 2) ---
    TARGET_POSTGRES_USER: str = Field(validation_alias='TARGET_POSTGRES_USER')
    TARGET_POSTGRES_PASSWORD: str = Field(validation_alias='TARGET_POSTGRES_PASSWORD')
    TARGET_POSTGRES_HOST: str = Field(validation_alias='TARGET_POSTGRES_HOST')
    TARGET_POSTGRES_PORT: int = Field(default=5432, validation_alias='TARGET_POSTGRES_PORT')
    TARGET_POSTGRES_DB: str = Field(validation_alias='TARGET_POSTGRES_DB')
    TARGET_POSTGRES_TABLE: str = Field(default="document_nlp_outputs", validation_alias='TARGET_POSTGRES_TABLE')

    # --- Service Logic & Scheduler ---
    NLP_BATCH_SIZE: int = Field(default=50, gt=0, validation_alias='NLP_BATCH_SIZE')
    SCHEDULER_INTERVAL_MINUTES: int = Field(default=15, gt=0, validation_alias='SCHEDULER_INTERVAL_MINUTES')
    MARK_AS_NLP_PROCESSED_IN_SOURCE_DB: bool = Field(default=True, validation_alias='MARK_AS_NLP_PROCESSED_IN_SOURCE_DB')

    @model_validator(mode='after')
    def derive_bertopic_model_path(cls, model: 'Settings') -> 'Settings':
        if model.BASE_DIR and model.BERTOPIC_MODEL_FILENAME:
            derived_path = os.path.join(model.BASE_DIR, model.BERTOPIC_MODEL_FILENAME)
            model.BERTOPIC_MODEL_PATH = derived_path # Overwrite the dummy default
        else:
            # This case should ideally not happen if BASE_DIR and FILENAME have defaults
            logger.error("Could not derive BERTOPIC_MODEL_PATH because BASE_DIR or BERTOPIC_MODEL_FILENAME is missing/empty.")
            # If it must be a string, and derivation fails, ensure it's still a string
            model.BERTOPIC_MODEL_PATH = "derivation_failed_path" 
        return model

    @property
    def postgres_dsn_asyncpg(self) -> str:
        return f"postgresql://{self.TARGET_POSTGRES_USER}:{self.TARGET_POSTGRES_PASSWORD}@{self.TARGET_POSTGRES_HOST}:{self.TARGET_POSTGRES_PORT}/{self.TARGET_POSTGRES_DB}"

    @field_validator("HEALTHCARE_SENTIMENT_LABELS", mode='before')
    @classmethod
    def parse_sentiment_labels(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if not v.strip():
                return ["Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"]
            try:
                parsed_list = json.loads(v)
                if not isinstance(parsed_list, list) or not all(isinstance(item, str) for item in parsed_list):
                    raise ValueError("HEALTHCARE_SENTIMENT_LABELS from env must be a JSON list of strings.")
                return parsed_list
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for HEALTHCARE_SENTIMENT_LABELS in env.")
        return ["Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

# --- Logging Setup ---
_log_level_to_use = settings.LOG_LEVEL.upper()
try:
    logger.remove() 
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(lambda msg: print(msg, end=""), format=log_format, level=_log_level_to_use, colorize=True)
    _is_loguru = True
except Exception: 
    import logging
    logging.basicConfig(
        level=_log_level_to_use,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True 
    )
    logger = logging.getLogger(__name__)
    _is_loguru = False

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("asyncpg").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

logger.info(f"Configuration loaded for {settings.SERVICE_NAME}. Log level: {_log_level_to_use}. Using {'Loguru' if _is_loguru else 'standard logging'}.")
logger.debug(f"BERTopic Model Filename: {settings.BERTOPIC_MODEL_FILENAME}")
logger.debug(f"Derived BERTOPIC_MODEL_PATH after validation: {settings.BERTOPIC_MODEL_PATH}")
logger.debug(f"Source PG: {settings.SOURCE_POSTGRES_DB}/{settings.SOURCE_POSTGRES_TABLE}")
logger.debug(f"Target PG: {settings.TARGET_POSTGRES_DB}/{settings.TARGET_POSTGRES_TABLE}")