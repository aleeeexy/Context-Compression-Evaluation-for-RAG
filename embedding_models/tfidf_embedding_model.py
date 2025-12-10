import numpy as np
import re
import math
from collections import Counter, defaultdict
from embedding_models.embedding_model import EmbeddingModel

class TfIdfEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "tfidf"):
        """Initialize the TF-IDF embedding model.
        
        Args:
            model_name: Name identifier for this model.
        """
        super().__init__(model_name)

        self.vocab = {} # term -> unique index
        self.documents = []
        self.idf_scores = {} # term -> idf score
        self.fitted = False # whether fit() is called
    
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _bigram_tokenize(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

    def _trigram_tokenize(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                for i in range(len(tokens) - 2)]

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
            bigrams = self._bigram_tokenize(doc)
            tokens.update(bigrams)
            trigrams = self._trigram_tokenize(doc)
            tokens.update(trigrams)
            for t in tokens:
                df[t] += 1

        N = len(self.documents)

        self.vocab = {term: idx for idx, term in enumerate(df.keys())}

        self.idf_scores = {
            term: math.log(N / (1 + df_val))
            for term, df_val in df.items()
        }

        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
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
        tokens += self._bigram_tokenize(text)
        tokens += self._trigram_tokenize(text)
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