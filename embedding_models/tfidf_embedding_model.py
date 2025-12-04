import numpy as np
import re
from Collections import Counter, defaultdict
from embedding_models.embedding_model import EmbeddingModel

class TfIdfEmbeddingModel(EmbeddingModel):
    def __init__(self):
        """
        Initialize the TF-IDF embedding model.
        """
        super().__init__()

        self.vocab = {} # term -> unique index
        self.documents = []
        self.idf_scores = {} # term -> idf score
        self.fitted = false # whether fit() is called
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, text: str):
        self.documents.append(text)

    def fit(self):
        """
        Build vocabulary, compute document frequencies (DF), and compute IDF.
        Required to calculate tf-idf embeddings
        """
        df = defaultdict(int)

        # calculate df (each token's frequency over all documents)
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] += 1

        N = len(self.documents)

        self.vocab = {term: idx for idx, term in enumerate(df.keys())}

        self.idf_scores = {
            term: math.log(N / (1 + df_val))
            for term, df_val in df.items()
        }

        self.fitted = True

    def embed(self, text: str) -> np.array[float]:
        """
        Generate TF-IDF embedding for the given text.

        Args:
            text (str): The input text to be embedded.

        Returns:
            vector of tf-idf embeddings for each term in vocabulary
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before embed().")

        tokens = self._tokenize(text)
        counter = Counter(tokens)

        total = sum(counter.values())

        # calculate tf (each token's frequency in the )
        tf = {t: c / total for t, c in counter.items()}

        vec = np.zeros(len(self.vocab), dtype=float)

        for term, tf_val in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = tf_val * self.idf_scores.get(term, 0.0)

        return vec