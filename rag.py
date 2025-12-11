import ollama

from document_parser.hotpotqa_loader import HotpotQALoader
from document_parser.document_parser import DocumentParser
from embedding_models.tfidf_embedding_model import TfIdfEmbeddingModel
from embedding_models.bert_embedding_model import BertEmbeddingModel
import numpy as np
import time

def tfidf_rag():
    """Run RAG using TF-IDF embeddings."""
    
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
        k = 10
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

def bert_rag():
    """Run RAG using BERT embeddings."""

    loader = HotpotQALoader()
    document_parser = loader.get_document_parser()
    documents = document_parser.get_documents()

    bert_embedding_model = BertEmbeddingModel("bert-model")
    for doc in documents:
        bert_embedding_model.add_document(doc.text)
        bert_embedding_model.add_document(documents[0].text)
    bert_embedding_model.fit()
    document_vectors = []
    for doc in documents:
        vec = bert_embedding_model.embed(doc.text)
        document_vectors.append(vec)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        question_vec = bert_embedding_model.embed(question)
        k = 10
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

def auto_rag_questions(qa_pairs: list[tuple[str, str]]):
    """Automatically ask questions using RAG with TF-IDF embeddings."""

    loader = HotpotQALoader()

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

    for question, answer in qa_pairs:
        question_vec = tfidf_embedding_model.embed(question)
        k = 5
        similarities = []
        for vec in document_vectors:
            sim = np.dot(question_vec, vec) / (np.linalg.norm(question_vec) * np.linalg.norm(vec))
            similarities.append(sim)
        top_k_indices = np.argsort(similarities)[-k:]
        context = f"<Context>\n" + "\n".join([documents[i].text for i in top_k_indices]) + f"\n</Context>"

        print(question)
        print("Expected Answer:", answer)
        print(query_tiny_lamma(context, question))
        print("=========")

def auto_ask_qeuestions_all_context(qa_pairs: list[tuple[str, str]]):
    """Automatically ask questions using RAG with all context provided."""

    loader = HotpotQALoader()

    document_parser = loader.get_document_parser()
    documents = document_parser.get_documents()
    full_context = "\n".join([doc.text for doc in documents])

    for question, answer in qa_pairs:
        print(question)
        print("Expected Answer:", answer)
        print(query_tiny_lamma(full_context, question))
        print("=========")

def auto_ask_questions_no_rag(qa_pairs: list[tuple[str, str]]):
    """Automatically ask questions without RAG."""

    for question, answer in qa_pairs:
        print(question)
        print("Expected Answer:", answer)
        print(query_tiny_lamma_no_context(question))
        print("=========")

def query_tiny_lamma(context, question):
    """Query TinyLlama with provided context and question."""

    response = ollama.chat("tinyllama", messages=[
        {"role": "system", "content": "You are a helpful assistant tasked with"
        " answering questions as best you can. You are given context to help you "
        " then a quesiton to answer. Do your best to answer the question based on your"
        " knowledge and the context provided. Answer very breifly, one word if possible."},
        {"role": "user", "content": context},
        {"role": "user", "content": question}
    ])
    return response["message"]["content"]

def query_tiny_lamma_no_context(question):
    """Query TinyLlama without any context."""

    response = ollama.chat("tinyllama", messages=[
        {"role": "system", "content": "You are a helpful assistant tasked with"
        " answering questions as best you can. Do your best to answer the question based on your"
        " knowledge. Answer very breifly, one word if possible."},
        {"role": "user", "content": question}
    ])
    return response["message"]["content"]

if __name__ == "__main__":
    # loader = HotpotQALoader()
    # print(len(loader.dataset))
    # qas = loader.get_questions_answers(10)
    # auto_rag_questions(qas)
    # start = time.time()
    # auto_rag_questions(qas)
    # end = time.time()
    # print("All context time taken:", end - start)
    tfidf_rag()
