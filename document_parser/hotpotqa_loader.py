from datasets import load_dataset
from .document_parser import DocumentParser
# from document_parser.document import Document

class HotpotQALoader:
    """Loader for the HotpotQA dataset."""
    def __init__(self, split: str = "validation", percentage: int = 10):
        self.dataset = load_dataset("hotpot_qa", "fullwiki", split=f"{split}[:{percentage}%]")

    def get_document_parser(self) -> DocumentParser:
        """Retrieve context documents from the HotpotQA dataset.

        Returns:
            list[str]: A list of context documents.
        """
        document_parser = DocumentParser()
        for example in self.dataset:
            context_sentences = example['context']['sentences']
            context_text = " ".join([" ".join(sentence) for sentence in context_sentences])
            document_parser.add_document(context_text)
        return document_parser
    
    def get_questions_answers(self, num: int = 10) -> list[tuple[str, str]]:
        """Retrieve questions from the HotpotQA dataset.

        Returns:
            list[str]: A list of questions.
        """
        qas = [(example['question'], example["answer"]) for example in self.dataset]
        return qas[:num]  # Return first num questions for brevity
    
    def get_questions(self) -> list[str]:
        """Retrieve questions from the HotpotQA dataset.

        Returns:
            list[str]: A list of questions.
        """
        questions = [example['question'] for example in self.dataset]
        return questions[:10]
    

loader = HotpotQALoader()
print(len(loader.dataset))
document_parser = loader.get_document_parser()
documents = document_parser.get_documents()
#for doc in documents[:2]:
#    print(doc.get_text()[:500])  # Print first 500 characters of each document
#print(len(documents))