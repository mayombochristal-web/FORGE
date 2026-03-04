import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dbddv01/gpt2-french-small"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
MAX_CONTEXT_TOKENS = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_NEW_TOKENS = 150

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.phi = {"phi_m": 0.6, "phi_c": 0.2, "phi_d": 0.2}

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.kb_texts = []
        self.text_hashes = set()

        self._emb_cache = {}
        self._response_cache = {}

        self.load_all()

    def _normalize_text(self, text):
        return " ".join(text.split())

    def _hash_text(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text, use_cache=True):
        if use_cache:
            key = hash(text)
            if key in self._emb_cache:
                return self._emb_cache[key]
        emb = self.embed_model.encode(text, normalize_embeddings=True)
        if use_cache:
            self._emb_cache[key] = emb
        return emb

    def evolve_phi(self, excitation):
        """Met à jour les métaphores en fonction de l'excitation (longueur de la question)."""
        self.phi["phi_m"] = min(0.9, max(0.4, self.phi["phi_m"] + excitation * 0.05))
        self.phi["phi_c"] = min(0.4, max(0.1, self.phi["phi_c"] + 0.01))
        self.phi["phi_d"] = min(0.3, max(0.05, self.phi["phi_d"] - excitation * 0.02))
        s = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= s

    def _get_dynamic_params(self):
        """
        Calcule les paramètres de génération en fonction des métaphores.
        - Température : liée à phi_d (bruit). Plus phi_d est faible, moins on veut de bruit.
          Formule : temp = 0.1 + 0.5 * phi_d   (borne entre 0.1 et 0.6)
        - repetition_penalty : liée à phi_c (cohérence). Plus phi_c est élevé, plus on pénalise.
          Formule : rep_penalty = 1.5 + phi_c   (borne entre 1.6 et 1.9)
        - top_p : lié à phi_m (mémoire). Plus phi_m est élevé, plus on est sélectif (top_p bas).
          Formule : top_p = 0.9 - 0.2 * phi_m   (borne entre 0.7 et 0.9)
        """
        temp = 0.1 + 0.5 * self.phi["phi_d"]
        rep_penalty = 1.5 + self.phi["phi_c"]
        top_p = 0.9 - 0.2 * self.phi["phi_m"]
        # Sécurisation des bornes
        temp = max(0.1, min(0.8, temp))
        rep_penalty = max(1.2, min(2.5, rep_penalty))
        top_p = max(0.6, min(0.95, top_p))
        return temp, rep_penalty, top_p

    def add_to_memory(self, text):
        start = 0
        text_len = len(text)
        added = 0
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk = text[start:end]
            normalized = self._normalize_text(chunk)
            if len(normalized) > 20 and normalized not in self.text_hashes:
                emb = self.get_embedding(chunk)
                self.index.add(np.array([emb], dtype=np.float32))
                self.kb_texts.append(chunk)
                self.text_hashes.add(normalized)
                added += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
        if added > 0:
            self.save_all()
        return added

    def search_memory(self, query, k=4):
        if self.index.ntotal == 0:
            return []
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k * 2)

        results = []
        seen = set()
        for idx in indices[0]:
            if idx != -1 and idx < len(self.kb_texts):
                text = self.kb_texts[idx]
                normalized = self._normalize_text(text)
                if normalized not in seen:
                    seen.add(normalized)
                    results.append(text)
                    if len(results) >= k:
                        break
        return results

    def generate_response(self, user_input, context_chunks=None, strict_mode=False):
        cache_key = self._hash_text(user_input + "".join(context_chunks or []) + str(strict_mode))
        if cache_key in self._response_cache:
            return self._response_cache[cache_key], context_chunks or []

        if context_chunks is None:
            context_chunks = self.search_memory(user_input)

        if strict_mode and not context_chunks:
            return "Information non trouvée dans les documents.", []

        # Construction du contexte
        max_chars = MAX_CONTEXT_TOKENS * 4
        context = ""
        for chunk in context_chunks:
            if len(context) + len(chunk) < max_chars:
                context += chunk + "\n"
            else:
                break
        context = context.strip()

        # Prompt simple
        if context:
            prompt = f"Contexte : {context}\n\nQuestion : {user_input}\nRéponse :"
        else:
            prompt = f"Question : {user_input}\nRéponse :"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)

        # Paramètres dynamiques basés sur les métaphores
        temp, rep_penalty, top_p = self._get_dynamic_params()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temp,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Réponse :" in full_response:
                response = full_response.split("Réponse :")[-1].strip()
            else:
                response = full_response.strip()
            response = self._clean_response(response)
        except Exception as e:
            response = f"Erreur de génération : {str(e)}"

        if not response or len(response) < 5:
            response = "Je n'ai pas d'information suffisante pour répondre."

        self._response_cache[cache_key] = response
        self.evolve_phi(min(1, len(user_input)/400))
        return response, context_chunks

    def _clean_response(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            text = ' '.join(sentences[:3]) + '...'
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return text
        return '\n'.join(lines[:3])

    def save_all(self):
        try:
            data = {
                "phi": self.phi,
                "kb": self.kb_texts,
                "hashes": list(self.text_hashes)
            }
            with open(self.mem_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Erreur sauvegarde mémoire : {e}")

    def load_all(self):
        if os.path.exists(self.mem_file):
            try:
                with open(self.mem_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.phi = data.get("phi", self.phi)
                self.kb_texts = data.get("kb", [])
                self.text_hashes = set(data.get("hashes", []))
                if self.kb_texts:
                    embs = self.embed_model.encode(self.kb_texts, normalize_embeddings=True)
                    self.index.reset()
                    self.index.add(np.array(embs, dtype=np.float32))
                    for txt, emb in zip(self.kb_texts, embs):
                        self._emb_cache[hash(txt)] = emb
            except Exception as e:
                print(f"Erreur chargement mémoire : {e}")
                self.kb_texts = []
                self.text_hashes = set()
                self.index.reset()