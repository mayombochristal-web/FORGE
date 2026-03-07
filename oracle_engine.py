import os
import numpy as np
import hashlib
import pickle

# ==========================================
# ORACLE ENGINE V18
# ==========================================

class OracleEngine:

    def __init__(self):

        self.memory = []
        self.vectors = []

        self.memory_file = "oracle_memory.pkl"

        self.load_memory()


    # ==========================================
    # VECTOR EMBEDDING SIMPLE
    # ==========================================

    def embed(self, text):

        words = text.lower().split()

        vec = np.zeros(128)

        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(),16)
            vec[h % 128] += 1

        return vec / (np.linalg.norm(vec) + 1e-9)


    # ==========================================
    # LOAD MEMORY (démarrage rapide)
    # ==========================================

    def load_memory(self):

        if os.path.exists(self.memory_file):

            with open(self.memory_file,"rb") as f:

                data = pickle.load(f)

                self.memory = data["memory"]
                self.vectors = data["vectors"]

        else:

            self.memory = []
            self.vectors = []


    # ==========================================
    # SAVE MEMORY
    # ==========================================

    def save_memory(self):

        data = {
            "memory": self.memory,
            "vectors": self.vectors
        }

        with open(self.memory_file,"wb") as f:
            pickle.dump(data,f)


    # ==========================================
    # ADD FILE MEMORY
    # ==========================================

    def add_file_memory(self, file):

        text = file.read().decode("utf-8",errors="ignore")

        chunks = text.split("\n")

        for chunk in chunks:

            if len(chunk.strip()) < 20:
                continue

            vec = self.embed(chunk)

            self.memory.append(chunk)
            self.vectors.append(vec)

        self.save_memory()

        return "Mémoire ajoutée"


    # ==========================================
    # VECTOR SEARCH
    # ==========================================

    def vector_search(self, query, k=5):

        q_vec = self.embed(query)

        scores = []

        for i,v in enumerate(self.vectors):

            sim = np.dot(q_vec,v)

            scores.append((sim,i))

        scores.sort(reverse=True)

        results = []

        for s,i in scores[:k]:

            results.append(self.memory[i])

        return results


    # ==========================================
    # TRANSFORMER ATTENTION (simplifié)
    # ==========================================

    def transformer_attention(self, query, contexts):

        attention_scores = []

        q_vec = self.embed(query)

        for ctx in contexts:

            v = self.embed(ctx)

            score = np.dot(q_vec,v)

            attention_scores.append((score,ctx))

        attention_scores.sort(reverse=True)

        return [c for s,c in attention_scores]


    # ==========================================
    # MULTI CONCEPT REASONING
    # ==========================================

    def multi_concept_reasoning(self, query, contexts):

        words = query.split()

        reasoning = []

        for ctx in contexts:

            score = sum(1 for w in words if w.lower() in ctx.lower())

            reasoning.append((score,ctx))

        reasoning.sort(reverse=True)

        return [c for s,c in reasoning]


    # ==========================================
    # GENERATE RESPONSE
    # ==========================================

    def generate_response(self, query):

        contexts = self.vector_search(query)

        attention = self.transformer_attention(query, contexts)

        reasoning = self.multi_concept_reasoning(query, attention)

        if reasoning:
            return reasoning[0]

        return "Je n'ai pas encore appris cela."
