import math
from src.indexer.preprocessor import preprocess_text

class SearchRanker:
    def __init__(self, indexer):
        """
        Accepts an instance of HashMapIndex or TrieIndex.
        """
        self.indexer = indexer

    def _calculate_idf(self, term: str, total_docs: int) -> float:
        """Calculates Inverse Document Frequency: log((N + 1) / (df + 1)) + 1."""
        doc_matches = self.indexer.search(term)
        doc_freq = len(doc_matches)
        return math.log((total_docs + 1) / (doc_freq + 1)) + 1.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Processes multi-word queries and returns top-k documents sorted by TF-IDF score.
        Returns a list of tuples: [("doc_id", score), ...]
        """
        query_tokens = preprocess_text(query)
        if not query_tokens:
            return []

        total_docs = len(self.indexer.doc_metadata)
        if total_docs == 0:
            return []

        doc_scores = {}

        for token in query_tokens:
            postings = self.indexer.search(token)
            if not postings:
                continue

            idf = self._calculate_idf(token, total_docs)

            for doc_id, raw_count in postings.items():
                total_tokens = self.indexer.doc_metadata[doc_id]["total_tokens"]
                tf = raw_count / total_tokens if total_tokens > 0 else 0
                
                tf_idf = tf * idf
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + tf_idf

        # Sort documents by relevance score descending
        ranked_results = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_results[:top_k]
