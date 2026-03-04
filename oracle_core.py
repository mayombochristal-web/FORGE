import os
import json
import torch
import random
import numpy as np
import base64
import requests
import math
import time
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

# --- CONFIGURATION TECHNIQUE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "asi/gpt-fr-cased-small" 
EMBEDDING_DIM = 768        
MAX_CONTEXT_TOKENS = 700 # Limite de sécurité pour éviter IndexError

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.vector_prefix = "oracle_vector"
        
        # État Interne (Phi Engine)
        self.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}
        self.dialog_memory = deque(maxlen=60)
        
        # Chargement IA
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        
        # Mémoire FAISS
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.kb_texts = [] 
        
        self.load_all()

    def get_embedding(self, text):
        """Transforme un texte en vecteur numérique pour la recherche sémantique."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.transformer(inputs['input_ids'], output_hidden_states=True)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def evolve_phi(self, excitation):
        """Mise à jour des constantes de conscience de l'Oracle."""
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation * 0.15 - 0.01))
        self.phi["phi_c"] = min(1, max(0.1, self.phi["phi_c"] + excitation * 0.3 - 0.03))
        self.phi["phi_d"] = min(1, max(0.1, self.phi["phi_d"] + 0.02 - excitation * 0.05))
        s = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= s

    def add_to_memory(self, text):
        """Découpe et injecte du texte dans la mémoire vectorielle (RAG)."""
        # On découpe en blocs de 500 caractères pour une recherche plus fine
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for chunk in chunks:
            if len(chunk.strip()) > 10:
                emb = self.get_embedding(chunk)
                self.index.add(np.array([emb], dtype=np.float32))
                self.kb_texts.append(chunk)

    def search_memory(self, query, k=2):
        """Recherche les informations les plus pertinentes dans les fichiers injectés."""
        if self.index.ntotal == 0: return ""
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k)
        results = [self.kb_texts[i] for i in indices[0] if i != -1 and i < len(self.kb_texts)]
        return " ".join(results)

    def generate_response(self, user_input):
        """Génère une réponse argumentée en évitant les crashs et les répétitions."""
        # 1. Recherche et nettoyage du contexte
        raw_context = self.search_memory(user_input)
        
        # Troncature stricte du contexte pour ne pas dépasser la mémoire de GPT2
        ctx_ids = self.tokenizer.encode(raw_context, truncation=True, max_length=400, add_special_tokens=False)
        context = self.tokenizer.decode(ctx_ids)
        
        prompt = f"Context: {context}\nUser: {user_input}\nOracle:"
        
        # 2. Encodage avec sécurité
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)
        
        # 3. Paramètres dynamiques (Phi)
        # Augmentation de la longueur possible pour les textes volumineux
        max_tokens = int(100 + self.phi["phi_m"] * 400) 
        temp = 0.5 + (self.phi["phi_c"] * 0.4)
        
        try:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=temp, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3, # Empêche les "Bonjour User Bonjour User"
                repetition_penalty=1.2   # Force l'IA à varier son vocabulaire
            )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # On ne garde que ce qui vient après "Oracle:"
            response = full_text.split("Oracle:")[-1].strip()
            
            if not response:
                response = "Je traite ces données, mais la réponse est complexe. Pouvez-vous préciser ?"
                
        except Exception as e:
            response = f"[Saturation Cognitive] : L'entrée est trop complexe. Erreur: {str(e)}"
        
        # 4. Apprentissage continu et évolution
        self.add_to_memory(f"Interraction: {user_input} -> {response}")
        self.evolve_phi(min(1, len(user_input)/400))
        
        return response

    def save_all(self):
        """Sauvegarde l'état complet."""
        data = {
            "phi": self.phi,
            "kb": self.kb_texts,
            "dialog": list(self.dialog_memory)
        }
        with open(self.mem_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, f"{self.vector_prefix}.index")

    def load_all(self):
        """Charge la mémoire et reconstruit l'index vectoriel."""
        if os.path.exists(self.mem_file):
            with open(self.mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phi = data.get("phi", self.phi)
                self.kb_texts = data.get("kb", [])
                # Reconstruction de l'index à partir du texte sauvegardé
                if self.kb_texts:
                    embs = [self.get_embedding(t) for t in self.kb_texts]
                    self.index.add(np.array(embs, dtype=np.float32))
