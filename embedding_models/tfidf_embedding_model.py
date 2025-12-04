from embedding_models.embedding_model import EmbeddingModel

class TfIdfEmbeddingModel(EmbeddingModel):
    def __init__(self):
        """
        Initialize the TF-IDF embedding model.
        """
        super().__init__()

        self.documents = []
        self.idf_scores = {}

    def embed(self, text: str) -> np.array[float]:
        """Generate TF-IDF embedding for the given text.

        Args:
            text (str): The input text to be embedded.
        """
        # Implementation for generating TF-IDF embeddings goes here
        
