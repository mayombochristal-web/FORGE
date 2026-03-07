import os
import json
from memory_storage import INDEX_PATH

def _load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    return {}

def _save_index(idx):
    with open(INDEX_PATH, "w", encoding="utf8") as f:
        json.dump(idx, f, ensure_ascii=False)

def update_index(text, filepath):
    idx = _load_index()
    words = set(text.lower().split())
    for w in words:
        if w not in idx:
            idx[w] = []
        if filepath not in idx[w]:
            idx[w].append(filepath)
    _save_index(idx)

def search_index(query_vector, max_candidates=20):
    idx = _load_index()
    words = list(query_vector.keys())
    paths = []
    for w in words:
        if w in idx:
            paths.extend(idx[w])
    unique_paths = list(set(paths))
    return unique_paths[:max_candidates]