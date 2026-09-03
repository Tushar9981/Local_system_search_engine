import pytest
from src.indexer.hash_index import HashMapIndex

def test_hash_map_indexing(tmp_path):
    
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Python search engine microproject python")
    
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("Privacy preserving local index")

    index = HashMapIndex()
    index.add_document("doc1", str(doc1))
    index.add_document("doc2", str(doc2))

    # Test O(1) keyword lookup
    python_results = index.search("python")
    assert python_results == {"doc1": 2}

    privacy_results = index.search("privacy")
    assert privacy_results == {"doc2": 1}

    # Test non-existent word
    assert index.search("java") == {}
