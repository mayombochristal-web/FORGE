# ============================================================
# ORACLE V15 Ω cosmos 
# Mini LLM Cognitif Expérimental
# ============================================================

import os
import sqlite3
import json
import math
import re
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle_vector_memory.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    text = text.lower()

    tokens = re.findall(r'\b[a-zàâéèêîôûç]+\b', text)

    return tokens


# ============================================================
# VECTOR EMBEDDING SIMPLE
# ============================================================

def text_vector(tokens):

    vector = defaultdict(int)

    for t in tokens:
        vector[t] += 1

    return vector


def cosine_similarity(v1, v2):

    intersection = set(v1.keys()) & set(v2.keys())

    num = sum(v1[x] * v2[x] for x in intersection)

    sum1 = sum(v**2 for v in v1.values())
    sum2 = sum(v**2 for v in v2.values())

    denom = math.sqrt(sum1) * math.sqrt(sum2)

    if denom == 0:
        return 0

    return float(num) / denom


# ============================================================
# VECTOR MEMORY DATABASE
# ============================================================

class VectorMemory:

    def __init__(self):

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

        self.create_table()

    def create_table(self):

        self.conn.execute("""

        CREATE TABLE IF NOT EXISTS memory(

        id INTEGER PRIMARY KEY,
        text TEXT,
        vector TEXT,
        source TEXT,
        date TEXT

        )

        """)

        self.conn.commit()

    def add(self, text, source="user"):

        tokens = tokenize(text)

        vector = text_vector(tokens)

        self.conn.execute(

            "INSERT INTO memory(text,vector,source,date) VALUES(?,?,?,?)",

            (
                text,
                json.dumps(vector),
                source,
                str(datetime.now())
            )

        )

        self.conn.commit()

    def search(self, query, top_k=5):

        tokens = tokenize(query)

        q_vector = text_vector(tokens)

        rows = self.conn.execute(

            "SELECT text,vector FROM memory"

        ).fetchall()

        results = []

        for text, vector_json in rows:

            vector = json.loads(vector_json)

            score = cosine_similarity(q_vector, vector)

            results.append((score, text))

        results.sort(reverse=True)

        return results[:top_k]

    def stats(self):

        count = self.conn.execute(

            "SELECT COUNT(*) FROM memory"

        ).fetchone()[0]

        return {"entries": count}


# ============================================================
# DOCUMENT SEGMENTATION
# ============================================================

def split_document(text, chunk_size=120):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


# ============================================================
# MULTI CONCEPT REASONER
# ============================================================

class MultiConceptReasoner:

    def extract_concepts(self, text):

        tokens = tokenize(text)

        concepts = []

        for t in tokens:

            if len(t) > 3:
                concepts.append(t)

        return list(set(concepts))

    def reasoning(self, question, memories):

        concepts = self.extract_concepts(question)

        links = []

        for m in memories:

            for c in concepts:

                if c in m.lower():

                    links.append((c, m))

        return links


# ============================================================
# QUESTION ANALYSIS
# ============================================================

class SpectralAnalysis:

    def analyze(self, question):

        tokens = tokenize(question)

        analysis = {

            "tokens": tokens,
            "longueur": len(tokens),
            "complexite": len(set(tokens)),
            "type": "information"

        }

        if "pourquoi" in tokens:

            analysis["type"] = "causal"

        if "comment" in tokens:

            analysis["type"] = "explication"

        return analysis


# ============================================================
# ANSWER GENERATOR
# ============================================================

class AnswerGenerator:

    def generate(self, question, memories, reasoning, analysis):

        answer = []

        answer.append("ANALYSE DE LA QUESTION")
        answer.append(json.dumps(analysis, indent=2))

        answer.append("\nCONNAISSANCES TROUVÉES")

        for m in memories:

            answer.append("- " + m)

        answer.append("\nRAISONNEMENT")

        for concept, mem in reasoning:

            answer.append(f"Concept '{concept}' relié à : {mem}")

        answer.append("\nCONCLUSION")

        if memories:

            answer.append(memories[0])

        else:

            answer.append(
                "Aucune connaissance pertinente trouvée dans la mémoire."
            )

        return "\n".join(answer)


# ============================================================
# DOCUMENT ANALYSIS REPORT
# ============================================================

def document_report(text):

    words = len(text.split())

    sentences = len(text.split("."))

    report = {

        "mots": words,
        "phrases": sentences,
        "longueur": len(text),
        "concepts_estimes": words // 6

    }

    return report


# ============================================================
# ORACLE BRAIN V16
# ============================================================

class OracleBrainV16:

    def __init__(self):

        self.memory = VectorMemory()

        self.reasoner = MultiConceptReasoner()

        self.analysis = SpectralAnalysis()

        self.generator = AnswerGenerator()

    # ========================================================
    # APPRENTISSAGE TEXTE
    # ========================================================

    def learn(self, text, source="user"):

        if len(text.strip()) > 10:

            self.memory.add(text, source)

    # ========================================================
    # APPRENTISSAGE DOCUMENT
    # ========================================================

    def learn_document(self, text, source="document"):

        chunks = split_document(text)

        for c in chunks:

            if len(c.strip()) > 30:

                self.learn(c, source)

        return len(chunks)

    # ========================================================
    # QUESTION
    # ========================================================

    def ask(self, question):

        analysis = self.analysis.analyze(question)

        results = self.memory.search(question)

        memories = [r[1] for r in results]

        reasoning = self.reasoner.reasoning(question, memories)

        answer = self.generator.generate(

            question,
            memories,
            reasoning,
            analysis
        )

        return answer

    # ========================================================
    # RAPPORT MEMOIRE
    # ========================================================

    def memory_report(self):

        stats = self.memory.stats()

        report = {

            "entrees_memoire": stats["entries"],
            "date": str(datetime.now())

        }

        return report


# ============================================================
# TEST LOCAL
# ============================================================

if __name__ == "__main__":

    oracle = OracleBrainV16()

    print("\nORACLE V15 Ω INITIALISÉ\n")

    while True:

        q = input("QUESTION : ")

        if q == "exit":

            break

        r = oracle.ask(q)

        print("\n", r)