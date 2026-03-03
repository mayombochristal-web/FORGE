# =====================================================
# 🧠 ORACLE Ω — SYSTÈME NEURONAL LINGUISTIQUE UNIFIÉ
# Compatible V6 + Omega GPT + GitHub Sync
# Version ultime : KV-cache, mémoire vectorielle, multilingue, fine-tuning en ligne
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
MODEL_NAME = "distilgpt2"  # Changez ici pour un modèle multilingue (ex: "dbmdz/german-gpt2", "microsoft/DialoGPT-medium")
EMBEDDING_DIM = 768  # À adapter selon le modèle (distilgpt2: 768, DialoGPT: 1024, etc.)
MAX_LEN = 512
LEARNING_RATE = 1e-5
MEMORY_SIZE = 10000  # Taille maximale de la mémoire vectorielle

# =====================================================
# MÉMOIRE VECTORIELLE (FAISS)
# =====================================================
class VectorMemory:
    def __init__(self, dim=EMBEDDING_DIM, max_size=MEMORY_SIZE):
        self.dim = dim
        self.max_size = max_size
        self.index = faiss.IndexFlatIP(dim)  # Inner product (similarité cosinus si vecteurs normalisés)
        self.texts = []          # Texte associé à chaque vecteur
        self.usage = []          # Compteur d'utilisation (pour élagage)
        self.embeddings = []     # Stockage local des vecteurs (pour récupération facile)

    def add(self, embedding, text):
        # Normalisation pour similarité cosinus
        embedding = embedding / np.linalg.norm(embedding)
        if len(self.texts) >= self.max_size:
            # Élagage : supprimer le moins utilisé
            min_usage_idx = np.argmin(self.usage)
            self._remove(min_usage_idx)
        self.index.add(np.array([embedding], dtype=np.float32))
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.usage.append(1)

    def _remove(self, idx):
        # Supprimer un élément de la mémoire (nécessite de reconstruire l'index)
        # Pour simplifier, on reconstruit l'index entier (peut être optimisé)
        self.index = faiss.IndexFlatIP(self.dim)
        new_embeddings = []
        new_texts = []
        new_usage = []
        for i, (emb, txt, cnt) in enumerate(zip(self.embeddments, self.texts, self.usage)):
            if i != idx:
                self.index.add(np.array([emb], dtype=np.float32))
                new_embeddings.append(emb)
                new_texts.append(txt)
                new_usage.append(cnt)
        self.embeddings = new_embeddings
        self.texts = new_texts
        self.usage = new_usage

    def search(self, query_emb, k=3):
        if self.index.ntotal == 0:
            return []
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores, indices = self.index.search(np.array([query_emb], dtype=np.float32), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.texts):
                self.usage[idx] += 1
                results.append(self.texts[idx])
        return results

# =====================================================
# 🧠 ORACLE BRAIN
# =====================================================
class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.model_file = memory_file.replace(".json", ".pt")
        self.dialog_memory = deque(maxlen=200)

        # --- Indicateurs cognitifs (phi) ---
        self.phi = {
            "Stabilité Ω": 0.5,
            "Plasticité": 0.5,
            "Mémoire": 0.5,
            "Attention": 0.5,
            "Perplexité": 0.0,
            "Tokens appris": 0
        }

        # --- Chargement du modèle et tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.train()  # mode train pour fine-tuning
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # --- Mémoire vectorielle ---
        self.vector_memory = VectorMemory()

        # --- Statistiques ---
        self.total_tokens_processed = 0
        self.running_loss = 0.0

        self._load_all()

    # =====================================================
    # TOKENISATION (via HF)
    # =====================================================
    def encode(self, text, return_tensors=True):
        tokens = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        if return_tensors:
            return tokens.to(DEVICE)
        else:
            return tokens[0].tolist()

    def decode(self, tensor):
        return self.tokenizer.decode(tensor[0], skip_special_tokens=True)

    # =====================================================
    # EXTRACTION D'EMBEDDING (dernière couche cachée)
    # =====================================================
    def get_embedding(self, text):
        inputs = self.encode(text)
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            # On prend la moyenne des embeddings de tous les tokens
            hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)
            emb = hidden.mean(dim=1).squeeze().cpu().numpy()
        return emb

    # =====================================================
    # APPRENTISSAGE ADAPTATIF (fine-tuning en ligne)
    # =====================================================
    def learn(self, text, reward=1.0):
        inputs = self.encode(text)
        if inputs.size(1) < 2:
            return

        # Forward
        outputs = self.model(inputs, labels=inputs)
        loss = outputs.loss

        # Backward avec récompense (le reward module le gradient)
        loss = loss * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour des indicateurs
        self.phi["Plasticité"] = min(1.0, self.phi["Plasticité"] + 0.01)
        self.phi["Stabilité Ω"] = max(0.0, 1 - loss.item())
        self.phi["Mémoire"] = min(1.0, len(self.vector_memory.texts) / MEMORY_SIZE)
        self.total_tokens_processed += inputs.size(1)
        self.phi["Tokens appris"] = self.total_tokens_processed
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss.item()
        self.phi["Perplexité"] = np.exp(self.running_loss)

        # Ajout à la mémoire vectorielle
        emb = self.get_embedding(text)
        self.vector_memory.add(emb, text)

    # =====================================================
    # GÉNÉRATION AUTOREGRESSIVE AVEC KV-CACHE
    # =====================================================
    def generate(self, prompt, max_new_tokens=50, temperature=0.8):
        self.model.eval()
        inputs = self.encode(prompt)
        past = None
        generated = inputs

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(generated, past_key_values=past, use_cache=True)
                logits = outputs.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)
                past = outputs.past_key_values

        response = self.decode(generated)
        # On ne garde que la partie générée après le prompt
        response = response[len(prompt):].strip()
        return response

    # =====================================================
    # PIPELINE PRINCIPAL (COMPATIBLE APP.PY)
    # =====================================================
    def process_input(self, user_input):
        self.dialog_memory.append(f"User: {user_input}")

        # --- Recherche en mémoire vectorielle pour enrichir le prompt ---
        emb = self.get_embedding(user_input)
        similar_texts = self.vector_memory.search(emb, k=2)
        context = " ".join(similar_texts) if similar_texts else ""
        enhanced_prompt = f"{context} {user_input}".strip()

        # --- Génération ---
        response = self.generate(enhanced_prompt)

        if not response.strip():
            response = "Je réfléchis encore à cette question."

        self.dialog_memory.append(f"Oracle: {response}")

        # --- Apprentissage (fine-tuning) sur l'échange complet ---
        full_exchange = f"{user_input} {response}"
        self.learn(full_exchange, reward=1.0)

        # --- Attention (simulée) ---
        self.phi["Attention"] = min(1.0, random.uniform(0.7, 1.0))

        self._save_all()
        return response

    # =====================================================
    # CYCLE DE SOMMEIL (appelé dans sidebar)
    # =====================================================
    def sleep_cycle(self):
        # Réduction de la mémoire dialogique
        if len(self.dialog_memory) > 50:
            self.dialog_memory = deque(list(self.dialog_memory)[-50:], maxlen=200)

        # Stabilisation des indicateurs
        for k in self.phi:
            self.phi[k] = max(0.3, self.phi[k] * 0.95)

        # Élagage de la mémoire vectorielle (on ne garde que les 1000 plus utilisés)
        if len(self.vector_memory.texts) > 1000:
            # Reconstruire l'index avec les textes les plus utilisés
            sorted_indices = np.argsort(self.vector_memory.usage)[-1000:]
            new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            new_texts = []
            new_usage = []
            new_embeddings = []
            for idx in sorted_indices:
                new_index.add(np.array([self.vector_memory.embeddings[idx]], dtype=np.float32))
                new_texts.append(self.vector_memory.texts[idx])
                new_usage.append(self.vector_memory.usage[idx])
                new_embeddings.append(self.vector_memory.embeddings[idx])
            self.vector_memory.index = new_index
            self.vector_memory.texts = new_texts
            self.vector_memory.usage = new_usage
            self.vector_memory.embeddings = new_embeddings

        self._save_all()

    # =====================================================
    # SAUVEGARDE / CHARGEMENT
    # =====================================================
    def _save_all(self):
        # Sauvegarde du modèle
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "phi": self.phi,
            "total_tokens": self.total_tokens_processed,
            "running_loss": self.running_loss
        }, self.model_file)

        # Sauvegarde de la mémoire dialogique (JSON)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(list(self.dialog_memory), f, ensure_ascii=False, indent=2)

        # Sauvegarde de la mémoire vectorielle (format FAISS + texte)
        # On sauvegarde l'index FAISS séparément
        faiss.write_index(self.vector_memory.index, self.memory_file.replace(".json", ".faiss"))
        with open(self.memory_file.replace(".json", "_vectors.json"), "w", encoding="utf-8") as f:
            json.dump({
                "texts": self.vector_memory.texts,
                "usage": self.vector_memory.usage
            }, f, ensure_ascii=False)

    def _load_all(self):
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.phi = checkpoint.get("phi", self.phi)
            self.total_tokens_processed = checkpoint.get("total_tokens", 0)
            self.running_loss = checkpoint.get("running_loss", 0.0)

        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                memory = json.load(f)
                self.dialog_memory = deque(memory, maxlen=200)

        # Chargement mémoire vectorielle
        faiss_file = self.memory_file.replace(".json", ".faiss")
        vectors_json = self.memory_file.replace(".json", "_vectors.json")
        if os.path.exists(faiss_file) and os.path.exists(vectors_json):
            self.vector_memory.index = faiss.read_index(faiss_file)
            with open(vectors_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.vector_memory.texts = data["texts"]
                self.vector_memory.usage = data["usage"]
                # Reconstruire la liste d'embeddings à partir de l'index (nécessite de les récupérer)
                # Pour simplifier, on recrée une liste vide, on les récupérera plus tard si besoin
                # Ici on va reconstruire les embeddings à partir de l'index (c'est possible via faiss)
                # Mais pour simplifier, on ne les charge pas, ils seront recalculés lors des ajouts
                self.vector_memory.embeddings = []
                # Optionnel : on pourrait parcourir l'index pour récupérer les vecteurs (coûteux)
                # On préfère les ignorer, ils seront recalculés si nécessaire