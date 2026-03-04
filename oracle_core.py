import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import hashlib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dbddv01/gpt2-french-small"               # Modèle génératif français
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Embedding multilingue
EMBEDDING_DIM = 384                                    # Dimension des embeddings
MAX_CONTEXT_TOKENS = 1024                               # Contexte maximum (1024 tokens pour GPT‑2)
CHUNK_SIZE = 800                                        # Taille des chunks (caractères)
CHUNK_OVERLAP = 100                                     # Chevauchement
DEFAULT_TEMPERATURE = 0.5                               # Température (un peu plus haute)
REPETITION_PENALTY = 1.8
TOP_P = 0.9
MAX_NEW_TOKENS = 250

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.phi = {"phi_m": 0.6, "phi_c": 0.2, "phi_d": 0.2}  # Métaphores internes

        # Modèle de génération
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

        # Modèle d'embedding
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))

        # Index FAISS
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.kb_texts = []          # Stockage des chunks
        self.text_hashes = set()     # Ensemble des hash pour déduplication

        # Cache pour les embeddings
        self._emb_cache = {}
        # Cache pour les réponses (évite de regénérer la même question)
        self._response_cache = {}

        self.load_all()

    def _normalize_text(self, text):
        """Normalise un texte pour la déduplication."""
        return " ".join(text.split())

    def _hash_text(self, text):
        """Hash rapide pour le cache des réponses."""
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text, use_cache=True):
        """Calcule l'embedding normalisé, avec cache."""
        if use_cache:
            key = hash(text)
            if key in self._emb_cache:
                return self._emb_cache[key]
        emb = self.embed_model.encode(text, normalize_embeddings=True)
        if use_cache:
            self._emb_cache[key] = emb
        return emb

    def evolve_phi(self, excitation):
        """Évolution des métaphores internes."""
        self.phi["phi_m"] = min(0.9, max(0.4, self.phi["phi_m"] + excitation * 0.05))
        self.phi["phi_c"] = min(0.4, max(0.1, self.phi["phi_c"] + 0.01))
        self.phi["phi_d"] = min(0.3, max(0.05, self.phi["phi_d"] - excitation * 0.02))
        s = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= s

    def add_to_memory(self, text):
        """
        Découpe le texte en chunks, évite les doublons, calcule les embeddings
        et les ajoute à l'index FAISS.
        """
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
        """
        Recherche les k chunks les plus pertinents, sans doublons.
        """
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
        """
        Génère une réponse à partir des chunks récupérés.
        Utilise un cache pour éviter de regénérer les mêmes questions.
        """
        # Création d'une clé de cache basée sur l'entrée et le contexte
        cache_key = self._hash_text(user_input + "".join(context_chunks or []) + str(strict_mode))
        if cache_key in self._response_cache:
            return self._response_cache[cache_key], context_chunks or []

        if context_chunks is None:
            context_chunks = self.search_memory(user_input)

        if strict_mode and not context_chunks:
            return "Information non trouvée dans les documents.", []

        # Construction du contexte avec limitation de taille
        max_chars = MAX_CONTEXT_TOKENS * 4  # approximation 1 token ~ 4 caractères
        context = ""
        for chunk in context_chunks:
            if len(context) + len(chunk) < max_chars:
                context += chunk + "\n---\n"
            else:
                break
        context = context.rstrip("\n---\n")

        # Instructions système améliorées
        if strict_mode:
            sys_instruction = (
                "Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni. "
                "Si la réponse ne se trouve pas dans le contexte, réponds exactement : "
                "'Information non trouvée dans les documents.'"
            )
        else:
            sys_instruction = (
                "Tu es un assistant documentaire. Réponds de façon précise et concise "
                "en t'appuyant d'abord sur le contexte. Si le contexte est insuffisant, "
                "tu peux compléter avec tes connaissances générales, mais reste factuel."
            )

        # Format de prompt structuré
        prompt = (
            f"### Instruction :\n{sys_instruction}\n\n"
            f"### Contexte :\n{context}\n\n"
            f"### Question :\n{user_input}\n\n"
            f"### Réponse :\n"
        )

        # Tokenisation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    do_sample=True,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extraction de la partie après "### Réponse :"
            if "### Réponse :" in full_response:
                response = full_response.split("### Réponse :")[-1].strip()
            else:
                response = full_response.strip()

            # Nettoyage : enlever les répétitions aberrantes et les fins de phrase coupées
            response = self._clean_response(response)
        except Exception as e:
            response = f"Erreur de génération : {str(e)}"

        # Mise en cache
        self._response_cache[cache_key] = response

        # Évolution des métaphores
        self.evolve_phi(min(1, len(user_input)/400))

        return response, context_chunks

    def _clean_response(self, text):
        """Nettoie la réponse : supprime les lignes vides multiples, limite aux premières phrases."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return text
        # Prendre les 5 premières lignes maximum
        cleaned = '\n'.join(lines[:5])
        # S'assurer que la phrase est complète (se termine par un point, etc.)
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '...'
        return cleaned

    def save_all(self):
        """Sauvegarde l'état (phi, texte des chunks, hash set) dans un fichier JSON."""
        data = {
            "phi": self.phi,
            "kb": self.kb_texts,
            "hashes": list(self.text_hashes)
        }
        with open(self.mem_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_all(self):
        """Charge l'état depuis le fichier JSON et reconstruit l'index FAISS."""
        if os.path.exists(self.mem_file):
            with open(self.mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phi = data.get("phi", self.phi)
                self.kb_texts = data.get("kb", [])
                self.text_hashes = set(data.get("hashes", []))
                if self.kb_texts:
                    # Recalcul des embeddings en batch
                    embs = self.embed_model.encode(self.kb_texts, normalize_embeddings=True)
                    self.index.add(np.array(embs, dtype=np.float32))
                    for txt, emb in zip(self.kb_texts, embs):
                        self._emb_cache[hash(txt)] = emb