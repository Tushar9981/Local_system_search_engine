from collections import defaultdict
from src.indexer.preprocessor import preprocess_text, extract_text_from_file

class HashMapIndex:
    def __init__(self):
      
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_metadata = {}

    def add_document(self, doc_id: str, file_path: str):
        """Extracts text from a file, preprocesses tokens, and indexes term frequencies."""
        raw_text = extract_text_from_file(file_path)
        tokens = preprocess_text(raw_text)
        
        self.doc_metadata[doc_id] = {
            "path": file_path,
            "total_tokens": len(tokens)
        }

        for token in tokens:
            self.index[token][doc_id] += 1

    def search(self, term: str) -> dict[str, int]:
        """Returns document IDs and term counts for a single word lookup in O(1) time."""
        term = term.lower().strip()
        return dict(self.index.get(term, {}))
