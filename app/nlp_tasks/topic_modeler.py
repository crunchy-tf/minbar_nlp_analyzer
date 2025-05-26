# nlp_analyzer_service/app/nlp_tasks/topic_modeler.py
from typing import List, Tuple, Optional, Dict, Any
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np # Import numpy for isinstance checks if probabilities are ndarray

from app.config import settings 

class TopicModeler:
    def __init__(self, model_path: str = settings.BERTOPIC_MODEL_PATH, sbert_model_name: str = settings.SBERT_MODEL_NAME):
        if not model_path or model_path == "dummy_path_will_be_overridden" or model_path == "derivation_failed_path":
            logger.critical(f"TopicModeler init: BERTOPIC_MODEL_PATH is invalid ('{model_path}'). BERTopic model will not be loaded.")
            self.model_path = None # Ensure path is None if invalid
        else:
            self.model_path = model_path
            
        self.sbert_model_name = sbert_model_name
        self.topic_model: Optional[BERTopic] = None
        self.embedding_model_instance: Optional[SentenceTransformer] = None

        self._load_embedding_model() # Load SBERT model first
        
        if self.model_path: # Only attempt to load BERTopic if path is valid
            self._load_model()           # Then load BERTopic model, passing the SBERT instance
        else:
            logger.error("Skipping BERTopic model loading due to invalid or unconfigured model_path.")


    def _load_embedding_model(self):
        """Loads the SentenceTransformer model."""
        if not self.sbert_model_name:
            logger.error("SBERT_MODEL_NAME not configured. Cannot load embedding model for TopicModeler.")
            return
        try:
            logger.info(f"Initializing SBERT embedding model: '{self.sbert_model_name}' for TopicModeler...")
            self.embedding_model_instance = SentenceTransformer(self.sbert_model_name)
            logger.info(f"SBERT embedding model '{self.sbert_model_name}' initialized successfully for TopicModeler.")
        except Exception as e:
            logger.error(f"Failed to initialize SBERT model '{self.sbert_model_name}' for TopicModeler: {e}", exc_info=True)
            self.embedding_model_instance = None

    def _load_model(self):
        """Loads the BERTopic model, providing the pre-loaded SBERT model."""
        if not self.embedding_model_instance:
            logger.error("SBERT embedding model instance is not available. Cannot effectively load BERTopic model for new predictions.")
            # If SBERT is critical, we should not proceed to load BERTopic as .transform will fail.
            return 
        
        # self.model_path should have been validated in __init__
        if not self.model_path:
             logger.error("BERTopic model_path is None or invalid. Cannot load BERTopic model.")
             return

        try:
            logger.info(f"Loading BERTopic model from: {self.model_path}")
            # Pass the pre-initialized SBERT model to BERTopic.load
            self.topic_model = BERTopic.load(
                self.model_path,
                embedding_model=self.embedding_model_instance 
            )
            logger.info("BERTopic model loaded successfully and SBERT embedding model explicitly assigned.")
            
            # +++ DETAILED DEBUGGING LOGS (as discussed) +++
            if self.topic_model:
                if hasattr(self.topic_model, 'embedding_model') and self.topic_model.embedding_model is not None:
                    logger.info(f"BERTopic has an embedding_model attribute. Type: {type(self.topic_model.embedding_model)}")
                    logger.info(f"Is it the same instance as self.embedding_model_instance? {self.topic_model.embedding_model is self.embedding_model_instance}")
                    logger.info(f"Does it have an 'encode' attribute? {hasattr(self.topic_model.embedding_model, 'encode')}")
                    if hasattr(self.topic_model.embedding_model, 'encode'):
                        logger.info(f"Is 'encode' callable? {callable(self.topic_model.embedding_model.encode)}")
                    
                    if isinstance(self.topic_model.embedding_model, SentenceTransformer):
                        logger.info("self.topic_model.embedding_model IS a SentenceTransformer instance.")
                    else:
                        logger.warning(f"self.topic_model.embedding_model IS NOT a SentenceTransformer instance. Actual type: {type(self.topic_model.embedding_model)}")
                else:
                    logger.warning("BERTopic model loaded, but its internal embedding_model attribute is missing or None.")
            else:
                logger.error("BERTopic model (self.topic_model) is None after load attempt.")
            # +++ END DETAILED DEBUGGING LOGS +++

        except FileNotFoundError:
            logger.error(f"BERTopic model file not found at {self.model_path}. Ensure it's copied into the Docker image correctly and the path is valid.")
            self.topic_model = None
        except Exception as e:
            logger.error(f"Failed to load BERTopic model from '{self.model_path}': {e}", exc_info=True)
            self.topic_model = None

    def get_topics_for_doc(self, document_text: str) -> Optional[Tuple[List[int], Optional[List[float]]]]:
        if not self.topic_model: 
            logger.error("BERTopic model not loaded. Cannot get topics.")
            return None
        
        # Check the embedding model within the loaded BERTopic model
        # This check is critical because BERTopic's .transform() relies on it.
        current_embedding_model = getattr(self.topic_model, 'embedding_model', None)
        if current_embedding_model is None:
            logger.error("BERTopic's internal embedding_model is missing (None). Cannot perform transform.")
            return None
        
        # Check if it has an 'encode' method (characteristic of SentenceTransformer)
        # Using hasattr first is safer than getattr with a default if unsure about 'encode' existence.
        if not hasattr(current_embedding_model, 'encode') or not callable(current_embedding_model.encode):
            logger.error(f"BERTopic's internal embedding_model (type: {type(current_embedding_model)}) does not have a callable 'encode' method. It might not be a functional SentenceTransformer. Cannot perform transform.")
            return None

        if not document_text or not document_text.strip(): 
            logger.warning("Empty input text for topic modeling. Assigning to outlier topic.")
            return ([-1], [1.0]) # Default outlier assignment for empty text
        
        try:
            logger.debug(f"Transforming document for topic modeling (first 100 chars): '{document_text[:100]}...'")
            topics, probabilities = self.topic_model.transform([document_text]) 
            logger.debug(f"BERTopic transform result - Topics: {topics}, Probabilities: {probabilities}")
            
            if topics is None: # Explicitly check if topics list itself is None
                logger.error("BERTopic transform returned None for topics list.")
                return None
            if not isinstance(topics, list) or (len(topics) > 0 and not isinstance(topics[0], int)):
                 logger.error(f"BERTopic transform returned unexpected type/content for topics: {type(topics)}, value: {topics}")
                 return None # Or handle appropriately (e.g., convert if possible, or log and return None)

            # Probabilities can also be None if calculate_probabilities=False (default for transform)
            # or if topics are outliers. Ensure it's a list of floats if not None.
            if probabilities is not None and not isinstance(probabilities, (list, np.ndarray)):
                logger.warning(f"BERTopic transform returned unexpected type for probabilities: {type(probabilities)}. Will attempt to use, but might fail later.")
            # If probabilities is an ndarray, it's fine as BERTopic often returns that.
            # If it's a list, ensure its elements are numbers or None.
            elif isinstance(probabilities, list):
                if not all(isinstance(p, (float, int, type(None))) or (isinstance(p, np.ndarray) and p.size==1) for p in probabilities): # Allow numpy floats/ints too
                    logger.warning(f"BERTopic transform returned list of probabilities with unexpected element types: {[type(p) for p in probabilities]}.")


            return topics, probabilities
        except Exception as e: 
            logger.error(f"Error caught during BERTopic transform for doc '{document_text[:50]}...': {e}", exc_info=True)
            return None
            
    def get_topic_details(self, topic_id: int) -> Optional[List[Tuple[str, float]]]:
        if not self.topic_model:
            logger.error("BERTopic model not loaded. Cannot get topic details.")
            return None
        try:
            return self.topic_model.get_topic(topic_id)
        except Exception as e:
            logger.error(f"Error getting details for topic ID {topic_id}: {e}", exc_info=True)
            return None
            
    def get_topic_name(self, topic_id: int) -> str:
        if not self.topic_model:
            return f"Topic {topic_id}" # Fallback
        try:
            topic_info_df = self.topic_model.get_topic_info(topic_id)
            if not topic_info_df.empty and 'Name' in topic_info_df.columns:
                name = topic_info_df['Name'].iloc[0]
                # A more robust check for custom name vs default BERTopic name
                is_default_name_format = name.startswith(str(topic_id) + "_") and all(part for part in name.split("_")[1:])
                if not is_default_name_format:
                    return name 
            
            details = self.get_topic_details(topic_id)
            if details:
                return f"{topic_id}_" + "_".join([word for word, score in details[:3]])
            return f"Topic {topic_id}" # Further fallback
        except Exception as e:
            logger.warning(f"Could not get custom name for topic {topic_id}: {e}")
            return f"Topic {topic_id}" # Final fallback

# Global instance (loaded at startup)
topic_model_instance = TopicModeler()

def get_topic_modeler_instance():
    return topic_model_instance