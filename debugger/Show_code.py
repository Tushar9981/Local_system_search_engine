import os
import time
import math
import psutil
import threading
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ==========================================
# 1. PREPROCESSOR & FILE SYSTEM MODULE
# ==========================================
def preprocess_text(text: str) -> list[str]:
    """Tokenizes text, converts to lowercase, and strips non-alphanumeric characters."""
    tokens = []
    for word in text.lower().split():
        cleaned = "".join(char for char in word if char.isalnum())
        if cleaned:
            tokens.append(cleaned)
    return tokens

def extract_text(file_path: str) -> str:
    """Reads content from text-readable files safely."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

# ==========================================
# 2. HASHMAP INDEX MODULE
# ==========================================
class HashMapIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_metadata = {}

    def add_document(self, doc_id: str, file_path: str, text: str):
        tokens = preprocess_text(text)
        self.doc_metadata[doc_id] = {"path": file_path, "total_tokens": len(tokens)}
        for token in tokens:
            self.index[token][doc_id] += 1

    def search(self, term: str) -> dict[str, int]:
        return dict(self.index.get(term.lower().strip(), {}))

# ==========================================
# 3. TRIE INDEX MODULE
# ==========================================
class TrieNode:
    __slots__ = ("children", "is_end", "postings")

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.postings = defaultdict(int)

class TrieIndex:
    def __init__(self):
        self.root = TrieNode()
        self.doc_metadata = {}

    def add_document(self, doc_id: str, file_path: str, text: str):
        tokens = preprocess_text(text)
        self.doc_metadata[doc_id] = {"path": file_path, "total_tokens": len(tokens)}
        for token in tokens:
            self._insert(token, doc_id)

    def _insert(self, token: str, doc_id: str):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.postings[doc_id] += 1

    def search(self, term: str) -> dict[str, int]:
        node = self._navigate(term.lower().strip())
        return dict(node.postings) if node and node.is_end else {}

    def prefix_search(self, prefix: str) -> dict[str, dict[str, int]]:
        prefix = prefix.lower().strip()
        node = self._navigate(prefix)
        if not node:
            return {}
        results = {}
        self._collect(node, prefix, results)
        return results

    def _navigate(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect(self, node: TrieNode, current: str, results: dict):
        if node.is_end:
            results[current] = dict(node.postings)
        for char, child in node.children.items():
            self._collect(child, current + char, results)

# ==========================================
# 4. TF-IDF RANKER MODULE
# ==========================================
class SearchRanker:
    def __init__(self, indexer):
        self.indexer = indexer

    def _calculate_idf(self, term: str, total_docs: int) -> float:
        doc_matches = self.indexer.search(term)
        doc_freq = len(doc_matches)
        return math.log((total_docs + 1) / (doc_freq + 1)) + 1.0

    def search(self, query: str, top_k: int = 15) -> list[tuple[str, float]]:
        tokens = preprocess_text(query)
        total_docs = len(self.indexer.doc_metadata)
        if not tokens or total_docs == 0:
            return []

        doc_scores = {}
        for token in tokens:
            postings = self.indexer.search(token)
            if not postings:
                continue
            idf = self._calculate_idf(token, total_docs)
            for doc_id, raw_count in postings.items():
                total_tokens = self.indexer.doc_metadata[doc_id]["total_tokens"]
                tf = raw_count / total_tokens if total_tokens > 0 else 0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (tf * idf)

        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ==========================================
# 5. TKINTER GUI INTERFACE FOR SPYDER
# ==========================================
class WindowsSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inverted Index Search & Ranker System")
        self.geometry("900x680")

        self.hash_idx = HashMapIndex()
        self.trie_idx = TrieIndex()
        self.ranker = None
        self.indexing_complete = False

        self._create_widgets()

    def _create_widgets(self):
        # Top Frame - Controls
        control_frame = ttk.LabelFrame(self, text=" Directory Indexer Controls ", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Target Directory:").grid(row=0, column=0, sticky="w")
        
        self.path_entry = ttk.Entry(control_frame, width=50)
        default_path = r"C:\Windows\System32" if os.name == "nt" else os.path.expanduser("~")
        self.path_entry.insert(0, default_path)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5)

        self.btn_index = ttk.Button(control_frame, text="Build Index", command=self._start_indexing_thread)
        self.btn_index.grid(row=0, column=2, padx=5, pady=5)

        self.status_label = ttk.Label(control_frame, text="Status: Ready", foreground="blue")
        self.status_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)

        # Middle Frame - Performance Metrics
        metrics_frame = ttk.LabelFrame(self, text=" Benchmark Metrics ", padding=10)
        metrics_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_hash_metrics = ttk.Label(metrics_frame, text="HashMap Index: Not built yet")
        self.lbl_hash_metrics.pack(anchor="w")

        self.lbl_trie_metrics = ttk.Label(metrics_frame, text="Trie Index: Not built yet")
        self.lbl_trie_metrics.pack(anchor="w")

        # Search Bar Frame
        search_frame = ttk.LabelFrame(self, text=" Search Interface ", padding=10)
        search_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(search_frame, text="Query (append '*' for prefix completion):").grid(row=0, column=0, sticky="w")
        self.search_entry = ttk.Entry(search_frame, width=45)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5)
        self.search_entry.bind("<Return>", lambda e: self._perform_search())

        self.btn_search = ttk.Button(search_frame, text="Search", command=self._perform_search, state="disabled")
        self.btn_search.grid(row=0, column=2, padx=5, pady=5)

        # Output Text Box Area
        results_frame = ttk.LabelFrame(self, text=" Ranked Search Results ", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_text = scrolledtext.ScrolledText(results_frame, wrap="word", font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True)

    def _start_indexing_thread(self):
        target_dir = self.path_entry.get().strip()
        if not os.path.exists(target_dir):
            messagebox.showerror("Error", f"Directory does not exist:\n{target_dir}")
            return

        self.btn_index.config(state="disabled")
        self.btn_search.config(state="disabled")
        self.status_label.config(text="Status: Crawling and Indexing files... Please wait.", foreground="orange")
        
        # Execute parsing in a daemon thread to prevent Spyder UI freezing
        threading.Thread(target=self._build_indexes, args=(target_dir,), daemon=True).start()

    def _build_indexes(self, target_dir: str):
        valid_extensions = {".txt", ".log", ".inf", ".ini", ".xml", ".cfg", ".py"}
        file_list = []

        # Walk system folders safely
        for root, _, files in os.walk(target_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    file_list.append(os.path.join(root, file))
                if len(file_list) >= 300:
                    break
            if len(file_list) >= 300:
                break

        if not file_list:
            self.after(0, self._indexing_failed, f"No readable text files found under '{target_dir}'.")
            return

        process = psutil.Process(os.getpid())

        # 1. Build HashMap Index
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()
        self.hash_idx = HashMapIndex()

        for idx, fpath in enumerate(file_list):
            content = extract_text(fpath)
            if content:
                self.hash_idx.add_document(f"doc_{idx+1}", fpath, content)

        t_hash_build = time.perf_counter() - t0
        m_hash = max(0.0, (process.memory_info().rss / (1024 * 1024)) - mem_before)

        # 2. Build Trie Index
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()
        self.trie_idx = TrieIndex()

        for idx, fpath in enumerate(file_list):
            content = extract_text(fpath)
            if content:
                self.trie_idx.add_document(f"doc_{idx+1}", fpath, content)

        t_trie_build = time.perf_counter() - t0
        m_trie = max(0.0, (process.memory_info().rss / (1024 * 1024)) - mem_before)

        self.ranker = SearchRanker(self.hash_idx)
        self.indexing_complete = True

        self.after(0, self._indexing_finished, len(file_list), t_hash_build, m_hash, t_trie_build, m_trie)

    def _indexing_failed(self, message: str):
        self.status_label.config(text="Status: Failed", foreground="red")
        self.btn_index.config(state="normal")
        messagebox.showwarning("Indexing Failed", message)

    def _indexing_finished(self, doc_count, t_hash, m_hash, t_trie, m_trie):
        self.status_label.config(text=f"Status: Done! Indexed {doc_count} system files.", foreground="green")
        self.btn_index.config(state="normal")
        self.btn_search.config(state="normal")

        self.lbl_hash_metrics.config(text=f"HashMap Index -> Duration: {t_hash:.4f}s | RAM Allocated: {m_hash:.2f} MB")
        self.lbl_trie_metrics.config(text=f"Trie Index    -> Duration: {t_trie:.4f}s | RAM Allocated: {m_trie:.2f} MB")

    def _perform_search(self):
        if not self.indexing_complete:
            return

        query = self.search_entry.get().strip()
        if not query:
            return

        self.result_text.delete("1.0", tk.END)

        if query.endswith("*"):
            prefix_term = query[:-1].strip()
            t0 = time.perf_counter()
            matches = self.trie_idx.prefix_search(prefix_term)
            latency = (time.perf_counter() - t0) * 1000

            self.result_text.insert(tk.END, f"=== Trie Prefix Search Results for '{prefix_term}*' ===\n")
            self.result_text.insert(tk.END, f"Latency: {latency:.3f} ms | Found {len(matches)} matching terms\n\n")

            if matches:
                for term, postings in list(matches.items())[:15]:
                    total_freq = sum(postings.values())
                    self.result_text.insert(tk.END, f" • Term: '{term:<20}' -> Appears {total_freq} times across {len(postings)} docs\n")
            else:
                self.result_text.insert(tk.END, "No terms found matching this prefix.\n")

        else:
            t0 = time.perf_counter()
            results = self.ranker.search(query, top_k=15)
            latency = (time.perf_counter() - t0) * 1000

            self.result_text.insert(tk.END, f"=== TF-IDF Ranked Search Results for '{query}' ===\n")
            self.result_text.insert(tk.END, f"Latency: {latency:.3f} ms\n\n")

            if results:
                for rank, (doc_id, score) in enumerate(results, start=1):
                    file_path = self.hash_idx.doc_metadata[doc_id]["path"]
                    self.result_text.insert(tk.END, f"Rank {rank:<2} | Score: {score:.6f} | ID: {doc_id}\n")
                    self.result_text.insert(tk.END, f"        Path: {file_path}\n\n")
            else:
                self.result_text.insert(tk.END, "No matching documents found.\n")

# Entry point for Spyder execution
if __name__ == "__main__":
    app = WindowsSearchApp()
    app.mainloop()
