from document import Document

class DocumentParser:

    def __init__(self):
        self.documents = []
        self.unique_words_set = set()

    def parse(self, file_path: str) -> list[Document]:
        """Parse the document at the given file path into a list of Document objects.

        Args:
            file_path (str): The path to the document file.

        Returns:
            list[Document]: A list of parsed Document objects.
        """
        pass

    def get_documents(self) -> list[Document]:
        """Retrieve the list of parsed Document objects.

        Returns:
            list[Document]: A list of parsed Document objects.
        """
        return self.documents

    def add_document(self, document_text: str) -> None:
        """Add a document to the list of parsed documents.

        Args:
            document (Document): The document to add.
        """
        self.documents.append(Document(document_text))

    # def unique_words() -> set[str]:
    #     """Retrieve a set of unique words from the parsed documents.

    #     Returns:
    #         set[str]: A set of unique words.
    #     """
    #     pass


