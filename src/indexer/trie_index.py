from collections import defaultdict
from src.indexer.preprocessor import preprocess_text, extract_text_from_file

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        
        self.postings = defaultdict(int)

class TrieIndex:
    def __init__(self):
        self.root = TrieNode()
        self.doc_metadata = {}

    def add_document(self, doc_id: str, file_path: str):
        """Extracts tokens and inserts each character into the prefix tree."""
        raw_text = extract_text_from_file(file_path)
        tokens = preprocess_text(raw_text)
        
        self.doc_metadata[doc_id] = {
            "path": file_path,
            "total_tokens": len(tokens)
        }

        for token in tokens:
            self._insert_token(token, doc_id)

    def _insert_token(self, token: str, doc_id: str):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.postings[doc_id] += 1

    def search(self, term: str) -> dict[str, int]:
        """Exact lookup for a word in O(L) time, where L is key length."""
        node = self._navigate_to_prefix(term.lower().strip())
        if node and node.is_end_of_word:
            return dict(node.postings)
        return {}

    def prefix_search(self, prefix: str) -> dict[str, dict[str, int]]:
        """Finds all words matching a given prefix and aggregates document counts."""
        prefix = prefix.lower().strip()
        node = self._navigate_to_prefix(prefix)
        if not node:
            return {}
        
        results = {}
        self._collect_words(node, prefix, results)
        return results

    def _navigate_to_prefix(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node: TrieNode, current_prefix: str, results: dict):
        if node.is_end_of_word:
            results[current_prefix] = dict(node.postings)
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_prefix + char, results)
