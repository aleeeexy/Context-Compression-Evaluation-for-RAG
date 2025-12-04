# create document parser
# pass in files
# pass documents into embedding model
# generate embeddings
# run on ollama
from document_parser.document_parser import DocumentParser
from document_parser.hotpotqa_loader import HotpotQALoader
from embedding_models.tfidf_embedding_model import TfIdfEmbeddingModel


loader = HotpotQALoader()
document_parser = loader.get_document_parser()
documents = document_parser.get_documents()
model = TfIdfEmbeddingModel("hi")
print("document: ", documents[0].text)
for doc in documents:
    model.add_document(doc.text)
model.add_document(documents[0].text)
model.fit()
vec = model.embed(loader.dataset[0]["question"])
print()
print("question: ", loader.dataset[0]["question"])

index_to_term = {idx: term for term, idx in model.vocab.items()}

vec = model.embed(loader.dataset[0]["question"])

print("question:", loader.dataset[0]["question"])
for idx, val in enumerate(vec):
    term = index_to_term[idx]
    if val > 0:
        print(f"{term}: {val:.4f}")