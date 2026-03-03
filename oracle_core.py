# =====================================================
# 🧠 ORACLE Ω — SYSTÈME NEURONAL LINGUISTIQUE UNIFIÉ
# Version stable avec HF Transformers, FAISS, et sauvegarde complète
# Compatible avec app.py existant
# =====================================================

import os
import json
import torch
import random
import numpy as np
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

# =====================================================
# CONFIGURATION
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilgpt2"  # Modèle léger, peut être changé (ex: "dbmdz/german-gpt2" pour allemand, "asi/gpt-fr-cased-small" pour français)
EMBEDDING_DIM = 768        # Correspond à distilgpt2, ajustez si vous changez de modèle
MAX_LEN = 512
MEMORY_SIZE = 10000        # Taille max de la mémoire vectorielle

# =====================================================
# MÉMOIRE VECTORIELLE (FAISS) avec persistance des embeddings
# =====================================================
class VectorMemory:
    def __init__(self, dim=EMBEDDING_DIM, max_size=MEMORY_SIZE):
        self.dim = dim
        self.max_size = max_size
        self.index = faiss.IndexFlatIP(dim)  # Similarité cosinus (après normalisation)
        self.texts = []
        self.usage = []
        self.embeddings = []  # On garde les vecteurs pour la sauvegarde

    def add(self, embedding, text):
        # Normalisation pour la similarité cosinus
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.zeros(self.dim)
        if len(self.texts) >= self.max_size:
            # Supprimer le moins utilisé
            min_usage_idx = np.argmin(self.usage)
            self._remove(min_usage_idx)
        self.index.add(np.array([embedding], dtype=np.float32))
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.usage.append(1)

    def _remove(self, idx):
        # Reconstruire l'index (simplifié, mais préserve les vecteurs)
        new_index = faiss.IndexFlatIP(self.dim)
        new_embeddings = []
        new_texts = []
        new_usage = []
        for i, (emb, txt, cnt) in enumerate(zip(self.embeddings, self.texts, self.usage)):
            if i != idx:
                new_index.add(np.array([emb], dtype=np.float32))
                new_embeddings.append(emb)
                new_texts.append(txt)
                new_usage.append(cnt)
        self.index = new_index
        self.embeddings = new_embeddings
        self.texts = new_texts
        self.usage = new_usage

    def search(self, query_emb, k=3):
        if self.index.ntotal == 0:
            return []
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        scores, indices = self.index.search(np.array([query_emb], dtype=np.float32), k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.texts):
                self.usage[idx] += 1
                results.append(self.texts[idx])
        return results

    def save(self, path_prefix):
        # Sauvegarde de l'index FAISS
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        # Sauvegarde des métadonnées et des embeddings
        with open(f"{path_prefix}_vectors.json", "w", encoding="utf-8") as f:
            json.dump({
                "texts": self.texts,
                "usage": self.usage,
                "embeddings": [emb.tolist() for emb in self.embeddings]
            }, f, ensure_ascii=False)

    def load(self, path_prefix):
        if os.path.exists(f"{path_prefix}.faiss") and os.path.exists(f"{path_prefix}_vectors.json"):
            self.index = faiss.read_index(f"{path_prefix}.faiss")
            with open(f"{path_prefix}_vectors.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.texts = data["texts"]
                self.usage = data["usage"]
                self.embeddings = [np.array(emb) for emb in data["embeddings"]]

# =====================================================
# 🧠 ORACLE BRAIN
# =====================================================
class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.model_file = memory_file.replace(".json", ".pt")
        self.vector_prefix = memory_file.replace(".json", "")  # pour les fichiers FAISS
        self.dialog_memory = deque(maxlen=200)

        # Indicateurs cognitifs (affichés dans la sidebar)
        self.phi = {
            "Stabilité": 0.5,
            "Plasticité": 0.5,
            "Mémoire": 0.5,
            "Attention": 0.5,
        }

        # Chargement du modèle et du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # pour le padding
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()  # Mode évaluation (pas de fine-tuning pour éviter les erreurs)

        # Mémoire vectorielle
        self.vector_memory = VectorMemory()
        self.total_tokens_processed = 0

        self._load_all()

    # ---------- Tokenisation ----------
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(DEVICE)

    def decode(self, tensor):
        return self.tokenizer.decode(tensor[0], skip_special_tokens=True)

    # ---------- Embedding (moyenne des dernières couches) ----------
    def get_embedding(self, text):
        inputs = self.encode(text)
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # dernière couche
            emb = hidden.mean(dim=1).squeeze().cpu().numpy()
        return emb

    # ---------- Génération avec cache automatique (HF generate) ----------
    def generate(self, prompt, max_new_tokens=50, temperature=0.8):
        inputs = self.encode(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_text = self.decode(outputs)
        # Retirer le prompt de la réponse
        response = full_text[len(prompt):].strip()
        if not response:
            response = "Je réfléchis encore à cette question."
        return response

    # ---------- Pipeline principal (appelé par app.py) ----------
    def process_input(self, user_input):
        # Sauvegarde du message utilisateur
        self.dialog_memory.append(f"User: {user_input}")

        # Recherche en mémoire vectorielle pour enrichir le contexte
        emb = self.get_embedding(user_input)
        similar_texts = self.vector_memory.search(emb, k=2)
        context = " ".join(similar_texts) if similar_texts else ""
        enhanced_prompt = f"{context} {user_input}".strip()

        # Génération de la réponse
        response = self.generate(enhanced_prompt)
        self.dialog_memory.append(f"Oracle: {response}")

        # Apprentissage : on stocke l'échange complet dans la mémoire vectorielle
        full_exchange = f"{user_input} {response}"
        emb_full = self.get_embedding(full_exchange)
        self.vector_memory.add(emb_full, full_exchange)

        # Mise à jour des indicateurs phi (simulée)
        self.phi["Attention"] = random.uniform(0.7, 1.0)
        self.phi["Mémoire"] = min(1.0, len(self.vector_memory.texts) / MEMORY_SIZE)
        self.phi["Plasticité"] = min(1.0, self.phi["Plasticité"] + 0.001)
        self.phi["Stabilité"] = max(0.0, self.phi["Stabilité"] - 0.001)

        # Sauvegarde
        self._save_all()
        return response

    # ---------- Cycle de sommeil (appelé depuis la sidebar) ----------
    def sleep_cycle(self):
        # Réduction de la mémoire dialogique
        if len(self.dialog_memory) > 50:
            self.dialog_memory = deque(list(self.dialog_memory)[-50:], maxlen=200)

        # Stabilisation des indicateurs
        for k in self.phi:
            self.phi[k] = max(0.3, self.phi[k] * 0.95)

        self._save_all()

    # ---------- Sauvegarde / Chargement ----------
    def _save_all(self):
        # Modèle (seulement l'état, pas nécessaire si on utilise le modèle pré-entraîné)
        # On sauvegarde les indicateurs et le compteur
        torch.save({
            "phi": self.phi,
            "total_tokens": self.total_tokens_processed
        }, self.model_file)

        # Mémoire dialogique (JSON)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(list(self.dialog_memory), f, ensure_ascii=False, indent=2)

        # Mémoire vectorielle
        self.vector_memory.save(self.vector_prefix)

    def _load_all(self):
        if os.path.exists(self.model_file):
            data = torch.load(self.model_file, map_location=DEVICE)
            self.phi = data.get("phi", self.phi)
            self.total_tokens_processed = data.get("total_tokens", 0)

        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.dialog_memory = deque(json.load(f), maxlen=200)

        self.vector_memory.load(self.vector_prefix)