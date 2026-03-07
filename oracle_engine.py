import os
import json
import math
from collections import Counter

# lecture fichiers
import pandas as pd
import PyPDF2
import docx


# ============================================================
# ORACLE V16 ENGINE
# ============================================================

class OracleEngine:

    def __init__(self):

        self.memory_file = "oracle_memory.json"

        self.memory_cache = []
        self.vector_index = []

        self.load_memory()

    # ========================================================
    # NETTOYAGE TEXTE
    # ========================================================

    def clean_text(self, text):

        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

        return text.strip()

    # ========================================================
    # VECTORISATION TEXTE
    # ========================================================

    def text_to_vector(self, text):

        words = text.lower().split()

        return Counter(words)

    # ========================================================
    # SIMILARITE COSINUS
    # ========================================================

    def cosine_similarity(self, v1, v2):

        intersection = set(v1.keys()) & set(v2.keys())

        numerator = sum(v1[x] * v2[x] for x in intersection)

        sum1 = sum(v1[x] ** 2 for x in v1)
        sum2 = sum(v2[x] ** 2 for x in v2)

        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # ========================================================
    # CHARGEMENT MEMOIRE
    # ========================================================

    def load_memory(self):

        if not os.path.exists(self.memory_file):

            self.memory_cache = []
            return

        try:

            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.memory_cache = json.load(f)

        except:
            self.memory_cache = []

        self.build_vector_index()

    # ========================================================
    # INDEX VECTORIEL
    # ========================================================

    def build_vector_index(self):

        self.vector_index = []

        for mem in self.memory_cache:

            text = mem["text"]

            vector = self.text_to_vector(text)

            self.vector_index.append({
                "text": text,
                "vector": vector
            })

    # ========================================================
    # RECHERCHE VECTORIELLE
    # ========================================================

    def vector_search(self, query):

        q_vector = self.text_to_vector(query)

        scores = []

        for mem in self.vector_index:

            score = self.cosine_similarity(q_vector, mem["vector"])

            scores.append((score, mem["text"]))

        scores.sort(key=lambda x: x[0], reverse=True)

        return scores[:5]

    # ========================================================
    # TRANSFORMER ATTENTION
    # ========================================================

    def transformer_attention(self, query, memories):

        words = query.lower().split()

        attention_scores = []

        for mem in memories:

            score = 0

            for w in words:

                if w in mem.lower():
                    score += 1

            attention_scores.append((score, mem))

        attention_scores.sort(reverse=True)

        return [m[1] for m in attention_scores]

    # ========================================================
    # RAISONNEMENT MULTI CONCEPT
    # ========================================================

    def multi_concept_reasoning(self, query, memories):

        concepts = query.split()

        results = []

        for mem in memories:

            score = 0

            for c in concepts:

                if c.lower() in mem.lower():
                    score += 1

            if score > 0:
                results.append(mem)

        return results

    # ========================================================
    # GENERATION REPONSE
    # ========================================================

    def generate_response(self, query, memories):

        if not memories:
            return "Je n'ai pas encore d'information sur ce sujet."

        response = "Voici ce que je sais :\n\n"

        for m in memories[:3]:

            response += "- " + m + "\n"

        return response

    # ========================================================
    # PIPELINE COMPLET
    # ========================================================

    def reason(self, query):

        vector_results = self.vector_search(query)

        memories = [m[1] for m in vector_results]

        memories = self.transformer_attention(query, memories)

        memories = self.multi_concept_reasoning(query, memories)

        return self.generate_response(query, memories)

    # ========================================================
    # AJOUT MEMOIRE TEXTE
    # ========================================================

    def add_memory(self, text):

        text = self.clean_text(text)

        entry = {"text": text}

        self.memory_cache.append(entry)

        with open(self.memory_file, "w", encoding="utf-8") as f:

            json.dump(self.memory_cache, f, indent=2)

        self.build_vector_index()

    # ========================================================
    # DECOUPAGE LONG TEXTE
    # ========================================================

    def split_text(self, text, size=500):

        chunks = []

        words = text.split()

        for i in range(0, len(words), size):

            chunk = " ".join(words[i:i+size])

            chunks.append(chunk)

        return chunks

    # ========================================================
    # LECTURE PDF
    # ========================================================

    def read_pdf(self, file):

        reader = PyPDF2.PdfReader(file)

        text = ""

        for page in reader.pages:

            text += page.extract_text() or ""

        return text

    # ========================================================
    # LECTURE WORD
    # ========================================================

    def read_docx(self, file):

        document = docx.Document(file)

        text = ""

        for p in document.paragraphs:

            text += p.text + "\n"

        return text

    # ========================================================
    # LECTURE EXCEL
    # ========================================================

    def read_excel(self, file):

        df = pd.read_excel(file)

        return df.to_string()

    # ========================================================
    # LECTURE CSV
    # ========================================================

    def read_csv(self, file):

        df = pd.read_csv(file)

        return df.to_string()

    # ========================================================
    # LECTURE GENERIQUE
    # ========================================================

    def read_file(self, uploaded_file):

        name = uploaded_file.name.lower()

        if name.endswith(".pdf"):
            return self.read_pdf(uploaded_file)

        if name.endswith(".docx"):
            return self.read_docx(uploaded_file)

        if name.endswith(".xlsx"):
            return self.read_excel(uploaded_file)

        if name.endswith(".csv"):
            return self.read_csv(uploaded_file)

        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        return ""

    # ========================================================
    # AJOUT MEMOIRE FICHIER
    # ========================================================

    def add_file_memory(self, uploaded_file):

        text = self.read_file(uploaded_file)

        if text.strip() == "":
            return "Fichier vide ou non supporté."

        chunks = self.split_text(text)

        for chunk in chunks:

            self.add_memory(chunk)

        return "Document ajouté à la mémoire."
