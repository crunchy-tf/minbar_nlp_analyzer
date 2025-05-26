# nlp_analyzer_service/app/config.py
import os
from typing import List, Union, Optional # Added Optional
from pydantic_settings import BaseSettings, SettingsConfigDict # For Pydantic V2
from pydantic import Field, field_validator, model_validator # <<< ADD model_validator
import json # For parsing list from env var
from loguru import logger # Assuming you want to use loguru here if available

class Settings(BaseSettings):
    SERVICE_NAME: str = "Minbar NLP Analyzer Service"
    LOG_LEVEL: str = "INFO"

    # --- Model Paths & Names ---
    BERTOPIC_MODEL_FILENAME: str = Field(default="bertopic_model_final_guided_multilang_gensim.joblib")
    SBERT_MODEL_NAME: str = Field(default="paraphrase-multilingual-mpnet-base-v2")
    SENTIMENT_MODEL_NAME: str = Field(default="joeddav/xlm-roberta-large-xnli")
    
    # BASE_DIR will point to /service because config.py is in /service/app/config.py
    # os.path.abspath(__file__) gives /service/app/config.py
    # os.path.dirname(os.path.abspath(__file__)) gives /service/app
    # os.path.dirname(os.path.dirname(os.path.abspath(__file__))) gives /service
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    BERTOPIC_MODEL_PATH: Optional[str] = Field(default=None) # Will be set by model_validator

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
            # Your Dockerfile copies the model to /service/model_filename.joblib
            # model.BASE_DIR will be /service (calculated from location of this config.py)
            # So this path will be /service/bertopic_model_final_guided_multilang_gensim.joblib
            derived_path = os.path.join(model.BASE_DIR, model.BERTOPIC_MODEL_FILENAME)
            model.BERTOPIC_MODEL_PATH = derived_path
            
            # Optional: Add a check here to see if the file exists at runtime start,
            # though the TopicModeler class will also check.
            # if not os.path.exists(model.BERTOPIC_MODEL_PATH):
            #     logger.warning(f"Derived BERTOPIC_MODEL_PATH does not exist: {model.BERTOPIC_MODEL_PATH}")
            # else:
            #     logger.info(f"BERTOPIC_MODEL_PATH derived: {model.BERTOPIC_MODEL_PATH}")
        elif model.BERTOPIC_MODEL_PATH is None: # Ensure it's not left as None if derivation fails
            logger.error("Could not derive BERTOPIC_MODEL_PATH because BASE_DIR or BERTOPIC_MODEL_FILENAME is missing.")
            # Potentially raise an error or set a dummy path to make Pydantic happy if it must be str
            # model.BERTOPIC_MODEL_PATH = "" # Or raise ValueError
        return model

    @property
    def postgres_dsn_asyncpg(self) -> str:
        # Uses TARGET_POSTGRES settings as this service WRITES to the target.
        # If source and target are same DB instance/db, these values will be the same.
        return f"postgresql://{self.TARGET_POSTGRES_USER}:{self.TARGET_POSTGRES_PASSWORD}@{self.TARGET_POSTGRES_HOST}:{self.TARGET_POSTGRES_PORT}/{self.TARGET_POSTGRES_DB}"

    @field_validator("HEALTHCARE_SENTIMENT_LABELS", mode='before')
    @classmethod
    def parse_sentiment_labels(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list): # If already a list (e.g., from default), return it
            return v
        if isinstance(v, str): # If from .env file (as a string)
            if not v.strip(): # Handle empty string from .env
                return ["Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"] # Default
            try:
                parsed_list = json.loads(v)
                if not isinstance(parsed_list, list) or not all(isinstance(item, str) for item in parsed_list):
                    raise ValueError("HEALTHCARE_SENTIMENT_LABELS from env must be a JSON list of strings.")
                return parsed_list
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for HEALTHCARE_SENTIMENT_LABELS in env.")
        # Fallback to default if input is neither list nor string (shouldn't happen with .env)
        return ["Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # For Pydantic V2, if you want to allow fields to be populated by validators like this,
        # you might need to set validate_assignment = True if you were assigning to self.BERTOPIC_MODEL_PATH
        # outside of a validator or __init__, but model_validator (mode='after') is the cleaner way.
    )

settings = Settings()

# --- Logging Setup (moved from Data Preprocessor's config, adjust if needed) ---
# This assumes you have loguru in requirements.txt. If not, it will use standard logging.
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
except Exception: # Catch broader exception if loguru setup fails for any reason
    import logging
    logging.basicConfig(
        level=_log_level_to_use,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True 
    )
    logger = logging.getLogger(__name__) # Re-get logger for this module after basicConfig
    _is_loguru = False

# Quieten noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING) # If you use httpx elsewhere
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("asyncpg").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING) # Transformers can be very verbose
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Add other libraries like NLTK, spaCy, Stanza if their DEBUG/INFO is too much
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # Often noisy from matplotlib via bertopic

logger.info(f"Configuration loaded for {settings.SERVICE_NAME}. Log level: {_log_level_to_use}. Using {'Loguru' if _is_loguru else 'standard logging'}.")
logger.debug(f"BERTopic Model Filename: {settings.BERTOPIC_MODEL_FILENAME}")
logger.debug(f"Derived BERTOPIC_MODEL_PATH: {settings.BERTOPIC_MODEL_PATH}") # Check if it's derived
logger.debug(f"Source PG: {settings.SOURCE_POSTGRES_DB}/{settings.SOURCE_POSTGRES_TABLE}")
logger.debug(f"Target PG: {settings.TARGET_POSTGRES_DB}/{settings.TARGET_POSTGRES_TABLE}")