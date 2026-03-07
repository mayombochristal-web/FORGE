import os
import json
from collections import defaultdict
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
    """Ajoute le fichier à l'index (mots -> liste de fichiers)"""
    idx = _load_index()
    words = set(text.lower().split())  # simplification
    for w in words:
        if w not in idx:
            idx[w] = []
        if filepath not in idx[w]:
            idx[w].append(filepath)
    _save_index(idx)

def search_index(query_vector, max_candidates=20):
    """Retourne une liste de chemins candidats pour la requête"""
    idx = _load_index()
    # Prend les mots de la requête (clés du vecteur)
    words = list(query_vector.keys())
    paths = []
    for w in words:
        if w in idx:
            paths.extend(idx[w])
    # Élimine les doublons et limite
    unique_paths = list(set(paths))
    return unique_paths[:max_candidates]