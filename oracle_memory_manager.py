import os
import json
import math
from memory_storage import save, load_all_metadata, count_memories
from memory_compressor import compress_old
from memory_index import update_index, search_index

class MemoryManager:
    def __init__(self):
        from memory_storage import init
        init()

    def store(self, text, vector, source):
        path = save(text, vector, source)
        update_index(text, path)
        if count_memories() > 100:
            compress_old()

    def search(self, query_vector, top_k=5):
        candidate_paths = search_index(query_vector, top_k * 2)
        results = []
        from memory_storage import load_vector
        for path in candidate_paths:
            vec, text = load_vector(path)
            if vec is None:
                continue
            score = self._cosine(query_vector, vec)
            if score > 0:
                results.append((score, text))
        results.sort(key=lambda x: -x[0])
        return results[:top_k]

    def all(self):
        return load_all_metadata()

    def count(self):
        return count_memories()

    def _cosine(self, v1, v2):
        inter = set(v1) & set(v2)
        num = sum(v1[x] * v2[x] for x in inter)
        s1 = sum(v**2 for v in v1.values())
        s2 = sum(v**2 for v in v2.values())
        denom = math.sqrt(s1) * math.sqrt(s2)
        if denom == 0:
            return 0
        return num / denom