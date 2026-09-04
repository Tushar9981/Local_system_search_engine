from src.indexer.hash_index import HashMapIndex
from src.search.ranker import SearchRanker

def test_tfidf_ranking(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Python search engine microproject python code")

    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("Privacy preserving local index with python")

    index = HashMapIndex()
    index.add_document("doc1", str(doc1))
    index.add_document("doc2", str(doc2))

    ranker = SearchRanker(index)
    results = ranker.search("python microproject")

    assert len(results) == 2
    assert results[0][0] == "doc1"
    assert results[0][1] > results[1][1]
