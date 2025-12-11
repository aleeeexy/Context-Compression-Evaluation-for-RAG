import numpy as np
from abc import ABC, abstractmethod
from document_parser.document import Document

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def __init__(self, model_name: str):
        """
        Initialize the embedding model with the given model name.
        Args:
            model_name (str): The name of the embedding model.
        """
        pass

    @abstractmethod
    def embed(self, document: Document) -> np.ndarray:
        """Generate embedding for the given text.

        Args:
            document (): The input text to be embedded.
        """
        pass
        