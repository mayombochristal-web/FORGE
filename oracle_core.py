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

# Configuration Technique
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilgpt2" 
EMBEDDING_DIM = 768        
MEMORY_SIZE = 10000        

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
        self.kb_texts = [] # Knowledge base textuelle
        
        self.load_all()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.transformer(inputs['input_ids'], output_hidden_states=True)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb

    def evolve_phi(self, excitation):
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation * 0.15 - 0.01))
        self.phi["phi_c"] = min(1, max(0.1, self.phi["phi_c"] + excitation * 0.3 - 0.03))
        self.phi["phi_d"] = min(1, max(0.1, self.phi["phi_d"] + 0.02 - excitation * 0.05))
        s = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= s

    def add_to_memory(self, text):
        emb = self.get_embedding(text)
        self.index.add(np.array([emb], dtype=np.float32))
        self.kb_texts.append(text)

    def search_memory(self, query, k=3):
        if self.index.ntotal == 0: return ""
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k)
        results = [self.kb_texts[i] for i in indices[0] if i != -1 and i < len(self.kb_texts)]
        return " ".join(results)

    def generate_response(self, user_input):
        # Récupération Contexte
        context = self.search_memory(user_input)
        prompt = f"Context: {context}\nUser: {user_input}\nOracle:"
        
        # Paramètres selon Phi
        max_tokens = int(50 + self.phi["phi_m"] * 450) # Capacité texte volumineux
        temp = 0.4 + (self.phi["phi_c"] * 0.6)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temp, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Oracle:")[-1].strip()
        
        # Apprentissage immédiat
        self.add_to_memory(f"Q: {user_input} A: {response}")
        self.evolve_phi(min(1, len(user_input)/300))
        
        return response

    def save_all(self):
        # Sauvegarde JSON simple pour compatibilité
        data = {
            "phi": self.phi,
            "kb": self.kb_texts,
            "dialog": list(self.dialog_memory)
        }
        with open(self.mem_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # Sauvegarde Index FAISS
        faiss.write_index(self.index, f"{self.vector_prefix}.index")

    def load_all(self):
        if os.path.exists(self.mem_file):
            with open(self.mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phi = data.get("phi", self.phi)
                self.kb_texts = data.get("kb", [])
                # Reconstruire l'index à partir de la KB
                for text in self.kb_texts:
                    emb = self.get_embedding(text)
                    self.index.add(np.array([emb], dtype=np.float32))
