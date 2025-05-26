# nlp_analyzer_service/app/nlp_tasks/topic_modeler.py
from typing import List, Tuple, Optional, Dict, Any
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.config import settings # CORRECTED IMPORT

class TopicModeler:
    def __init__(self, model_path: str = settings.BERTOPIC_MODEL_PATH, sbert_model_name: str = settings.SBERT_MODEL_NAME): # CORRECTED DEFAULTS
        if not model_path or model_path == "dummy_path_will_be_overridden" or model_path == "derivation_failed_path":
            logger.critical(f"TopicModeler init: BERTOPIC_MODEL_PATH is invalid ('{model_path}'). Cannot load BERTopic model.")
            self.model_path = None
        else:
            self.model_path = model_path
        self.sbert_model_name = sbert_model_name
        self.topic_model: Optional[BERTopic] = None
        self.embedding_model_instance: Optional[SentenceTransformer] = None
        self._load_embedding_model()
        if self.model_path: self._load_model()
        else: logger.error("Skipping BERTopic model loading due to invalid model_path.")

    def _load_embedding_model(self):
        if not self.sbert_model_name: logger.error("SBERT_MODEL_NAME not configured."); return
        try:
            logger.info(f"Initializing SBERT embedding model: '{self.sbert_model_name}' for TopicModeler...")
            self.embedding_model_instance = SentenceTransformer(self.sbert_model_name)
            logger.info(f"SBERT embedding model '{self.sbert_model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SBERT model '{self.sbert_model_name}': {e}", exc_info=True)
            self.embedding_model_instance = None

    def _load_model(self):
        if not self.embedding_model_instance: logger.error("SBERT embedding model instance not available for TopicModeler."); return
        if not self.model_path: logger.error("BERTopic model path invalid. Cannot load model."); return # Added check
        try:
            logger.info(f"Loading BERTopic model from: {self.model_path}")
            self.topic_model = BERTopic.load(self.model_path, embedding_model=self.embedding_model_instance)
            logger.info("BERTopic model loaded successfully and SBERT embedding model explicitly assigned.")
            if self.topic_model and not self.topic_model.embedding_model: logger.warning("BERTopic model loaded, but internal embedding_model is not set.")
            elif self.topic_model and self.topic_model.embedding_model: logger.info("Verified: BERTopic model's internal embedding_model is set.")
        except FileNotFoundError: logger.error(f"BERTopic model file not found at {self.model_path}.") ; self.topic_model = None
        except Exception as e: logger.error(f"Failed to load BERTopic model from '{self.model_path}': {e}", exc_info=True); self.topic_model = None

    def get_topics_for_doc(self, document_text: str) -> Optional[Tuple[List[int], Optional[List[float]]]]:
        if not self.topic_model: logger.error("BERTopic model not loaded."); return None
        if not hasattr(self.topic_model, 'embedding_model') or not self.topic_model.embedding_model: logger.error("BERTopic embedding_model missing."); return None
        if not callable(getattr(self.topic_model.embedding_model, "encode", None)): logger.error("BERTopic embedding_model not functional."); return None
        if not document_text or not document_text.strip(): logger.warning("Empty input for topic modeling."); return ([-1], [1.0])
        try:
            logger.debug(f"Transforming doc for topics: '{document_text[:50]}...'")
            topics, probabilities = self.topic_model.transform([document_text])
            return topics, probabilities
        except Exception as e: logger.error(f"Error in BERTopic transform: {e}", exc_info=True); return None
            
    def get_topic_details(self, topic_id: int) -> Optional[List[Tuple[str, float]]]:
        if not self.topic_model: logger.error("BERTopic model not loaded."); return None
        try: return self.topic_model.get_topic(topic_id)
        except Exception as e: logger.error(f"Error getting details for topic ID {topic_id}: {e}", exc_info=True); return None
            
    def get_topic_name(self, topic_id: int) -> str:
        if not self.topic_model: return f"Topic {topic_id}"
        try:
            topic_info_df = self.topic_model.get_topic_info(topic_id)
            if not topic_info_df.empty and 'Name' in topic_info_df.columns:
                name = topic_info_df['Name'].iloc[0]
                if not (name.startswith(str(topic_id) + "_") and len(name.split("_")) > 1): return name 
            details = self.get_topic_details(topic_id)
            return f"{topic_id}_" + "_".join([word for word, score in details[:3]]) if details else f"Topic {topic_id}"
        except Exception as e: logger.warning(f"Could not get custom name for topic {topic_id}: {e}"); return f"Topic {topic_id}"

topic_model_instance = TopicModeler()
def get_topic_modeler_instance(): return topic_model_instance