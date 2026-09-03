from src.indexer.preprocessor import preprocess_text

def test_preprocess_lowercasing_and_punctuation():
    tokens = preprocess_text("Hello, World! Python3 Searching...")
    assert tokens == ["hello", "world", "python3", "searching"]

def test_preprocess_stop_words_filtering():
    tokens = preprocess_text("The quick brown fox is on the log")
    assert tokens == ["quick", "brown", "fox", "log"]

def test_empty_string():
    assert preprocess_text("") == []
