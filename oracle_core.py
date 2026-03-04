import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dbddv01/gpt2-french-small"          # Modèle génératif français plus robuste
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Embedding multilingue
EMBEDDING_DIM = 384                               # Dimension des embeddings
MAX_CONTEXT_TOKENS = 750                           # Taille max du prompt
CHUNK_SIZE = 800                                   # Taille des chunks (caractères)
CHUNK_OVERLAP = 100                                 # Chevauchement
DEFAULT_TEMPERATURE = 0.3                           # Température fixe (un peu plus haute pour éviter la monotonie)

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.phi = {"phi_m": 0.6, "phi_c": 0.2, "phi_d": 0.2}  # Métaphores internes
        
        # Modèle de génération
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        
        # Modèle d'embedding dédié
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
        
        # Index FAISS
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.kb_texts = []          # Stockage des chunks
        self.text_hashes = set()     # Ensemble des hash des textes normalisés (pour déduplication)
        
        # Cache pour les embeddings
        self._emb_cache = {}
        
        self.load_all()

    def _normalize_text(self, text):
        """Normalise un texte pour la déduplication (supprime espaces multiples, strip)."""
        return " ".join(text.split())

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
        """Évolution des métaphores internes (gardée pour compatibilité)."""
        self.phi["phi_m"] = min(0.9, max(0.4, self.phi["phi_m"] + excitation * 0.05))
        self.phi["phi_c"] = min(0.4, max(0.1, self.phi["phi_c"] + 0.01))
        self.phi["phi_d"] = min(0.3, max(0.05, self.phi["phi_d"] - excitation * 0.02))
        s = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= s

    def add_to_memory(self, text):
        """
        Découpe le texte en chunks avec chevauchement, évite les doublons,
        calcule les embeddings et les ajoute à l'index FAISS.
        """
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk = text[start:end]
            normalized = self._normalize_text(chunk)
            if len(normalized) > 20 and normalized not in self.text_hashes:
                # Nouveau chunk
                emb = self.get_embedding(chunk)
                self.index.add(np.array([emb], dtype=np.float32))
                self.kb_texts.append(chunk)
                self.text_hashes.add(normalized)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        # Sauvegarde automatique après ajout
        self.save_all()

    def search_memory(self, query, k=4):
        """
        Recherche les k chunks les plus pertinents, sans doublons.
        Retourne une liste de textes uniques.
        """
        if self.index.ntotal == 0:
            return []
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k * 2)  # On en demande plus pour filtrer
        
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
        - strict_mode : si True, refuse de répondre si aucune source pertinente.
        """
        if context_chunks is None:
            context_chunks = self.search_memory(user_input)
        
        # Si mode strict et pas de contexte, réponse immédiate
        if strict_mode and not context_chunks:
            return "Information non trouvée dans les documents.", []
        
        # Construction du contexte (concaténation avec séparateur)
        if context_chunks:
            # Limiter la longueur totale du contexte pour ne pas dépasser la capacité du modèle
            # On va mesurer en tokens approximativement (car le tokenizer est lent, on utilise une règle empirique)
            max_chars = MAX_CONTEXT_TOKENS * 4  # approximation grossière (1 token ~ 4 caractères en français)
            context = ""
            for chunk in context_chunks:
                if len(context) + len(chunk) < max_chars:
                    context += chunk + "\n---\n"
                else:
                    break
            context = context.rstrip("\n---\n")
        else:
            context = ""
        
        # Instruction système
        if strict_mode:
            sys_instruction = "Réponds de manière concise en te basant uniquement sur la source. Si la source ne contient pas la réponse, dis : 'Information non trouvée dans les documents.'"
        else:
            sys_instruction = "Réponds de manière concise en te basant sur la source. Si la source manque d'informations, tu peux utiliser tes connaissances générales, mais reste factuel."
        
        prompt = (
            f"SYSTÈME: {sys_instruction}\n"
            f"SOURCE: {context}\n"
            f"QUESTION: {user_input}\n"
            f"RÉPONSE ANALYTIQUE:"
        )
        
        # Tokenisation et troncature
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=DEFAULT_TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.8,           # Pénalité renforcée
                    no_repeat_ngram_size=3,            # Empêche les répétitions de 3-grammes
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extraire la partie après "RÉPONSE ANALYTIQUE:"
            if "RÉPONSE ANALYTIQUE:" in response:
                response = response.split("RÉPONSE ANALYTIQUE:")[-1].strip()
            else:
                response = response.strip()
        except Exception as e:
            response = f"Erreur de génération : {str(e)}"
        
        # Mise à jour des métaphores
        self.evolve_phi(min(1, len(user_input)/400))
        
        return response, context_chunks

    def save_all(self):
        """Sauvegarde l'état (phi, texte des chunks, hash set) dans un fichier JSON."""
        # Convertir le set en liste pour JSON
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
                    # Remplir le cache (optionnel)
                    for txt, emb in zip(self.kb_texts, embs):
                        self._emb_cache[hash(txt)] = emb