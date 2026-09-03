from src.indexer.trie_index import TrieIndex

def test_trie_exact_search(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Privacy preserving inverted index engine")

    index = TrieIndex()
    index.add_document("doc1", str(doc1))

    assert index.search("privacy") == {"doc1": 1}
    assert index.search("engine") == {"doc1": 1}
    assert index.search("missing") == {}

def test_trie_prefix_search(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("test testing tester automated tests")

    index = TrieIndex()
    index.add_document("doc1", str(doc1))

    prefix_matches = index.prefix_search("test")
    
    assert "test" in prefix_matches
    assert "testing" in prefix_matches
    assert "tester" in prefix_matches
    assert "tests" in prefix_matches
