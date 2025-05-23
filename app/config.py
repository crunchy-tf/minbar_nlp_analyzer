# nlp_analyzer_service/app/config.py
import os
from typing import List, Union # Added Union
from pydantic_settings import BaseSettings, SettingsConfigDict # For Pydantic V2
from pydantic import Field, field_validator # For Pydantic V2
import json # For parsing list from env var

class Settings(BaseSettings):
    SERVICE_NAME: str = "Minbar NLP Analyzer Service"
    LOG_LEVEL: str = "INFO"

    # --- Model Paths & Names ---
    BERTOPIC_MODEL_FILENAME: str = Field(default="bertopic_model_final_guided_multilang_gensim.joblib")
    SBERT_MODEL_NAME: str = Field(default="paraphrase-multilingual-mpnet-base-v2")
    SENTIMENT_MODEL_NAME: str = Field(default="joeddav/xlm-roberta-large-xnli")
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BERTOPIC_MODEL_PATH: str # Will be derived

    HEALTHCARE_SENTIMENT_LABELS: List[str] = [
        "Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"
    ]
    TOP_N_KEYWORDS: int = 10

    # --- Source DB (Data Preprocessor's Output in minbar_processed_data) ---
    SOURCE_POSTGRES_USER: str
    SOURCE_POSTGRES_PASSWORD: str
    SOURCE_POSTGRES_HOST: str
    SOURCE_POSTGRES_PORT: int = Field(default=5432)
    SOURCE_POSTGRES_DB: str
    SOURCE_POSTGRES_TABLE: str = Field(default="processed_documents")
    NLP_ANALYZER_STATUS_FIELD_IN_SOURCE: str = Field(default="nlp_analyzer_v1_status")

    # --- Target DB (This service's output table, in the SAME DB as source for Option 2) ---
    TARGET_POSTGRES_USER: str
    TARGET_POSTGRES_PASSWORD: str
    TARGET_POSTGRES_HOST: str
    TARGET_POSTGRES_PORT: int = Field(default=5432)
    TARGET_POSTGRES_DB: str
    TARGET_POSTGRES_TABLE: str = Field(default="document_nlp_outputs")

    # --- Service Logic & Scheduler ---
    NLP_BATCH_SIZE: int = Field(default=50, gt=0)
    SCHEDULER_INTERVAL_MINUTES: int = Field(default=15, gt=0)
    MARK_AS_NLP_PROCESSED_IN_SOURCE_DB: bool = Field(default=True)

    def __init__(self, **values: any): # Add __init__ to derive BERTOPIC_MODEL_PATH
        super().__init__(**values)
        self.BERTOPIC_MODEL_PATH = os.path.join(self.BASE_DIR, self.BERTOPIC_MODEL_FILENAME)

    @property
    def postgres_dsn_asyncpg(self) -> str:
        # For Option 2, source and target DB use the same DSN if they are the same instance/db
        # Ensure user/pass/host/port/db for target match source in .env if they are the same
        return f"postgresql://{self.TARGET_POSTGRES_USER}:{self.TARGET_POSTGRES_PASSWORD}@{self.TARGET_POSTGRES_HOST}:{self.TARGET_POSTGRES_PORT}/{self.TARGET_POSTGRES_DB}"

    # Example for parsing HEALTHCARE_SENTIMENT_LABELS if it were from .env as a JSON string
    @field_validator("HEALTHCARE_SENTIMENT_LABELS", mode='before')
    @classmethod
    def parse_sentiment_labels(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed_list = json.loads(v)
                if not isinstance(parsed_list, list) or not all(isinstance(item, str) for item in parsed_list):
                    raise ValueError("HEALTHCARE_SENTIMENT_LABELS must be a JSON list of strings.")
                return parsed_list
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for HEALTHCARE_SENTIMENT_LABELS.")
        return ["Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"] # Default fallback

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()