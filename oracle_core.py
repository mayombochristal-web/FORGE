import os
import json
import torch
import numpy as np
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "asi/gpt-fr-cased-small" 
EMBEDDING_DIM = 768        
MAX_CONTEXT_TOKENS = 750 

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.vector_prefix = "oracle_vector"
        self.phi = {"phi_m": 0.6, "phi_c": 0.2, "phi_d": 0.2} # Priorité à la Mémoire (M)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.kb_texts = [] 
        self.load_all()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.transformer(inputs['input_ids'], output_hidden_states=True)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def evolve_phi(self, excitation):
        # Évolution plus lente pour garder la stabilité documentaire
        self.phi["phi_m"] = min(0.9, max(0.4, self.phi["phi_m"] + excitation * 0.05))
        self.phi["phi_c"] = min(0.4, max(0.1, self.phi["phi_c"] + 0.01))
        self.phi["phi_d"] = min(0.3, max(0.05, self.phi["phi_d"] - excitation * 0.02))
        s = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= s

    def add_to_memory(self, text):
        chunks = [text[i:i+600] for i in range(0, len(text), 600)]
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                emb = self.get_embedding(chunk)
                self.index.add(np.array([emb], dtype=np.float32))
                self.kb_texts.append(chunk)

    def search_memory(self, query, k=4): # K=4 pour plus de contexte
        if self.index.ntotal == 0: return ""
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k)
        return " ".join([self.kb_texts[i] for i in indices[0] if i != -1 and i < len(self.kb_texts)])

    def generate_response(self, user_input):
        context = self.search_memory(user_input)
        
        # Prompt structuré pour la fidélité
        prompt = (
            f"SYSTÈME: Réponds de manière concise en te basant uniquement sur la source.\n"
            f"SOURCE: {context}\n"
            f"QUESTION: {user_input}\n"
            f"RÉPONSE ANALYTIQUE:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)
        
        # Température basse (0.2 - 0.4) pour éviter les hallucinations
        temp = 0.2 + (self.phi["phi_d"] * 0.3)
        
        try:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=250, 
                temperature=temp, 
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.25,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("RÉPONSE ANALYTIQUE:")[-1].strip()
        except Exception as e:
            response = f"Erreur de génération : {str(e)}"
        
        self.evolve_phi(min(1, len(user_input)/400))
        return response

    def save_all(self):
        data = {"phi": self.phi, "kb": self.kb_texts}
        with open(self.mem_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_all(self):
        if os.path.exists(self.mem_file):
            with open(self.mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phi = data.get("phi", self.phi)
                self.kb_texts = data.get("kb", [])
                if self.kb_texts:
                    embs = [self.get_embedding(t) for t in self.kb_texts]
                    self.index.add(np.array(embs, dtype=np.float32))
