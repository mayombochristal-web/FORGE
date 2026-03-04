import os
import json
import torch
import numpy as np
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

# --- CONFIGURATION TECHNIQUE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "asi/gpt-fr-cased-small" 
EMBEDDING_DIM = 768        
MAX_CONTEXT_TOKENS = 700 

class OracleBrain:
    def __init__(self, mem_file="oracle_memory.json"):
        self.mem_file = mem_file
        self.vector_prefix = "oracle_vector"
        
        # État Interne (Phi Engine)
        self.phi = {"phi_m": 0.33, "phi_c": 0.33, "phi_d": 0.34}
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
        """Transforme un texte en vecteur numérique."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.transformer(inputs['input_ids'], output_hidden_states=True)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def evolve_phi(self, excitation):
        """Mise à jour des constantes de conscience."""
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation * 0.15 - 0.01))
        self.phi["phi_c"] = min(1, max(0.1, self.phi["phi_c"] + excitation * 0.3 - 0.03))
        self.phi["phi_d"] = min(1, max(0.1, self.phi["phi_d"] + 0.02 - excitation * 0.05))
        s = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= s

    def add_to_memory(self, text):
        """Injection RAG."""
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for chunk in chunks:
            if len(chunk.strip()) > 10:
                emb = self.get_embedding(chunk)
                self.index.add(np.array([emb], dtype=np.float32))
                self.kb_texts.append(chunk)

    def search_memory(self, query, k=3):
        """Recherche sémantique."""
        if self.index.ntotal == 0: return ""
        emb = self.get_embedding(query)
        scores, indices = self.index.search(np.array([emb], dtype=np.float32), k)
        results = [self.kb_texts[i] for i in indices[0] if i != -1 and i < len(self.kb_texts)]
        return " ".join(results)

    def generate_response(self, user_input):
        """Génère une réponse avec synthèse et remise en question."""
        raw_context = self.search_memory(user_input, k=3)
        ctx_ids = self.tokenizer.encode(raw_context, truncation=True, max_length=450, add_special_tokens=False)
        context = self.tokenizer.decode(ctx_ids)
        
        prompt = (
            f"Source Documentaire: {context}\n"
            f"Analyse Utilisateur: {user_input}\n"
            f"Oracle (Réflexion et Synthèse):"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)
        dynamic_temp = 0.3 + (self.phi["phi_d"] * 0.5) 
        
        try:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=int(150 + self.phi["phi_m"] * 350), 
                temperature=dynamic_temp, 
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text.split("Oracle (Réflexion et Synthèse):")[-1].strip()
            
            if self.phi["phi_c"] > 0.7 and "Cependant" not in response:
                response += "\n\nNote : Cette analyse se base strictement sur les fragments identifiés, mais l'interprétation globale pourrait varier selon le prisme sociolinguistique adopté."

        except Exception as e:
            response = f"[Alerte de saturation] : Erreur technique: {str(e)}"
        
        self.evolve_phi(min(1, len(user_input)/350))
        return response

    def save_all(self):
        """Sauvegarde persistante."""
        data = {"phi": self.phi, "kb": self.kb_texts}
        with open(self.mem_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, f"{self.vector_prefix}.index")

    def load_all(self):
        """Chargement et reconstruction de l'index."""
        if os.path.exists(self.mem_file):
            try:
                with open(self.mem_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.phi = data.get("phi", self.phi)
                    self.kb_texts = data.get("kb", [])
                if self.kb_texts:
                    embs = [self.get_embedding(t) for t in self.kb_texts]
                    self.index.add(np.array(embs, dtype=np.float32))
            except: pass
