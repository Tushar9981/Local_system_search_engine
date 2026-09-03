import re
from pypdf import PdfReader

DEFAULT_STOP_WORDS = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "is", "of", "for", "with"}

def extract_text_from_file(file_path: str) -> str:
    """Reads plain text (.txt, .md) or .pdf files."""
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def preprocess_text(text: str, stop_words: set = DEFAULT_STOP_WORDS) -> list[str]:
    """Converts text to lowercase, strips non-alphanumeric noise, and filters stop-words."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [token for token in tokens if token not in stop_words]
