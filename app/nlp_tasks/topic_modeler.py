# FILE: app/nlp_tasks/topic_modeler.py
# Topic Modeler Logic

# Standard library imports
from typing import List, Tuple, Optional, Dict, Any

# Third-party imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer # <<< ADDED IMPORT
from loguru import logger
# import joblib # Not strictly needed if using BERTopic.load(), but BERTopic might use it internally

# Application-specific imports
from app.config import BERTOPIC_MODEL_PATH, SBERT_MODEL_NAME # <<< ADDED SBERT_MODEL_NAME

class TopicModeler:
    def __init__(self, model_path: str = BERTOPIC_MODEL_PATH, sbert_model_name: str = SBERT_MODEL_NAME):
        self.model_path = model_path
        self.sbert_model_name = sbert_model_name # Store SBERT model name
        self.topic_model: Optional[BERTopic] = None
        self.embedding_model_instance: Optional[SentenceTransformer] = None # To hold the SBERT instance

        self._load_embedding_model() # Load SBERT model first
        self._load_model()           # Then load BERTopic model, passing the SBERT instance

    def _load_embedding_model(self):
        """Loads the SentenceTransformer model."""
        if not self.sbert_model_name:
            logger.error("SBERT_MODEL_NAME not configured. Cannot load embedding model.")
            return
        try:
            logger.info(f"Initializing SBERT embedding model: '{self.sbert_model_name}' for TopicModeler...")
            # This will download from Hugging Face Hub if not cached by transformers/sentence-transformers
            # The cache location is typically ~/.cache/huggingface/hub or ~/.cache/torch/sentence_transformers
            # Inside Docker, if TRANSFORMERS_CACHE or HF_HOME is set, it will use that.
            self.embedding_model_instance = SentenceTransformer(self.sbert_model_name)
            logger.info(f"SBERT embedding model '{self.sbert_model_name}' initialized successfully for TopicModeler.")
        except Exception as e:
            logger.error(f"Failed to initialize SBERT model '{self.sbert_model_name}' for TopicModeler: {e}", exc_info=True)
            self.embedding_model_instance = None

    def _load_model(self):
        """Loads the BERTopic model, providing the pre-loaded SBERT model."""
        if not self.embedding_model_instance:
            logger.error("SBERT embedding model instance not available. BERTopic model loading might result in a non-functional model for new predictions.")
            # We can still attempt to load the BERTopic structure, but .transform() will fail.
            # Alternatively, make this a hard failure:
            # self.topic_model = None
            # return 

        try:
            logger.info(f"Loading BERTopic model from: {self.model_path}")
            # Pass the pre-initialized SBERT model to BERTopic.load
            # This ensures that the loaded BERTopic model has a functional embedding model,
            # especially if it was saved with `save_embedding_model=False`.
            self.topic_model = BERTopic.load(
                self.model_path,
                embedding_model=self.embedding_model_instance 
            )
            logger.info("BERTopic model loaded successfully and SBERT embedding model explicitly assigned.")
            
            # Sanity check after loading
            if self.topic_model and not self.topic_model.embedding_model:
                logger.warning("BERTopic model loaded, but its internal embedding_model attribute is still not set. Predictions might fail.")
            elif self.topic_model and self.topic_model.embedding_model:
                logger.info("Verified: BERTopic model's internal embedding_model is now set.")

        except FileNotFoundError:
            logger.error(f"BERTopic model file not found at {self.model_path}. Ensure it's copied into the Docker image correctly at the expected path.")
            self.topic_model = None
        except Exception as e:
            logger.error(f"Failed to load BERTopic model: {e}", exc_info=True)
            self.topic_model = None

    def get_topics_for_doc(self, document_text: str) -> Optional[Tuple[List[int], Optional[List[float]]]]:
        if not self.topic_model:
            logger.error("BERTopic model not loaded. Cannot get topics.")
            return None
        
        # Explicitly check if the embedding model within BERTopic is functional
        if not hasattr(self.topic_model, 'embedding_model') or not self.topic_model.embedding_model:
            logger.error("BERTopic model is loaded, but its internal embedding_model is missing or not initialized. Cannot perform transform.")
            return None
        # Further check if it's an actual model object capable of embedding
        if not callable(getattr(self.topic_model.embedding_model, "embed_documents", None)) and \
           not callable(getattr(self.topic_model.embedding_model, "encode", None)): # SBERT uses .encode
            logger.error("BERTopic model's embedding_model is not a functional SentenceTransformer object. Cannot perform transform.")
            return None

        if not document_text or not document_text.strip():
            logger.warning("Input document for topic modeling is empty. Assigning to outlier topic.")
            return ([-1], [1.0]) # Default outlier assignment

        try:
            logger.debug(f"Transforming document for topic modeling: '{document_text[:100]}...'")
            topics, probabilities = self.topic_model.transform([document_text])
            logger.debug(f"Assigned topics: {topics}, Probabilities: {probabilities}")
            return topics, probabilities
        except AttributeError as ae: # Catch specific 'NoneType' object has no attribute 'embed_documents' or similar
            logger.error(f"AttributeError during BERTopic transform (likely embedding model issue): {ae}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error during BERTopic transform for doc '{document_text[:100]}...': {e}", exc_info=True)
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
            return f"Topic {topic_id}"
        try:
            # Attempt to get custom name first
            topic_info_df = self.topic_model.get_topic_info(topic_id) # Get info for specific topic
            if not topic_info_df.empty and 'Name' in topic_info_df.columns:
                name = topic_info_df['Name'].iloc[0]
                # Check if the name is not the default BERTopic format (ID_kw1_kw2...)
                # This indicates an LLM or custom name was set.
                if not (name.startswith(str(topic_id) + "_") and len(name.split("_")) > 1):
                    return name 
            
            # Fallback to generating name from keywords if custom name isn't good or not present
            details = self.get_topic_details(topic_id)
            if details:
                return f"{topic_id}_" + "_".join([word for word, score in details[:3]]) # Default format
            return f"Topic {topic_id}" # Absolute fallback

        except Exception as e:
            logger.warning(f"Could not get custom name for topic {topic_id}: {e}")
            return f"Topic {topic_id}" # Fallback

# Global instance (loaded at startup)
topic_model_instance = TopicModeler()

def get_topic_modeler_instance():
    return topic_model_instance