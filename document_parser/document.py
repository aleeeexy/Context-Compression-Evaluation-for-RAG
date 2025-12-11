class Document:
    def __init__(self, document_text: str):
        """Initialize a Document with the given text."""
        self.text = document_text

    def get_text(self) -> str:
        """Retrieve the text content of the document."""
        return self.text

    # def word_frequency(self) -> dict[str, int]:
    #         words = self.content.split()
    #         frequency = {}
    #         for word in words:
    #             word = word.lower()
    #             frequency[word] = frequency.get(word, 0) + 1
    #         return frequency
        
    # def unique_words(self) -> set[str]:
    #     return self.word_frequency_map.keys()