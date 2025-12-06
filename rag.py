import ollama

from document_parser.hotpotqa_loader import HotpotQALoader
from document_parser.document_parser import DocumentParser
from embedding_models.tfidf_embedding_model import TfIdfEmbeddingModel
import numpy as np

def tfidf_rag():
    loader = HotpotQALoader()
    print(loader.get_questions())
    document_parser = loader.get_document_parser()
    documents = document_parser.get_documents()

    tfidf_embedding_model = TfIdfEmbeddingModel("tfidf-model")
    for doc in documents:
        tfidf_embedding_model.add_document(doc.text)
        tfidf_embedding_model.add_document(documents[0].text)
    tfidf_embedding_model.fit()

    document_vectors = []
    for doc in documents:
        vec = tfidf_embedding_model.embed(doc.text)
        document_vectors.append(vec)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        question_vec = tfidf_embedding_model.embed(question)
        k = 5
        #write logic to find the k most similar documents using cosine similarity
        similarities = []
        for vec in document_vectors:
            sim = np.dot(question_vec, vec) / (np.linalg.norm(question_vec) * np.linalg.norm(vec))
            similarities.append(sim)
        top_k_indices = np.argsort(similarities)[-k:]
        context = f"<Context>\n" + "\n".join([documents[i].text for i in top_k_indices]) + f"\n</Context>"
        # print("\nContext retrieved for RAG:\n", context)

        print("context")
        for idx in top_k_indices:
            print(f"Document {idx} (similarity: {similarities[idx]:.4f}):")
            print(documents[idx].text[:50])  # Print first 50 characters of each document
            print("-----")
        print(query_tiny_lamma(context, question))


def query_tiny_lamma(context, question):
    response = ollama.chat("tinyllama", messages=[
        {"role": "system", "content": "You are a helpful assistant tasked with"
        " answering questions as best you can. You are given context to help you "
        " then a quesiton to answer. Do your best to answer the question based on your"
        " knowledge and the context provided."},
        {"role": "user", "content": context},
        {"role": "user", "content": question}
    ])
    return response["message"]["content"]

# response = ollama.chat("tinyllama", messages=[
#     {"role": "system", "content": "You are a helpful assistant tasked with"
#     " answering questions as best you can. You are given context to help you "
#     " then a quesiton to answer. Do your best to answer the question based on your"
#     " knowledge and the context provided."},
#     {"role": "user", "content": context},
#     {"role": "user", "content": question}
# ])

if __name__ == "__main__":

    tfidf_rag()