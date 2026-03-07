# ============================================================
# ORACLE V15 Ω COSMOS
# Moteur Cognitif Expérimental
# ============================================================

import os
import sqlite3
import json
import re
import math
from datetime import datetime
from collections import defaultdict, Counter

# ============================================================
# CONFIG
# ============================================================

MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "cosmos_memory.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

# ============================================================
# TOKENIZER PROPRE
# ============================================================

STOPWORDS = {
    "les","des","une","dans","pour","avec",
    "qui","que","est","sur","pas","plus",
    "par","comme","mais","donc","car",
    "nous","vous","ils","elle","elles",
    "aux","ces","ses","leur","leurs"
}

def tokenize(text):

    text = text.lower()

    tokens = re.findall(r"[a-zàâéèêëîïôûùüç]{3,}", text)

    tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens

# ============================================================
# VECTOR EMBEDDING SIMPLE
# ============================================================

def text_vector(tokens):

    vector = defaultdict(int)

    for token in tokens:
        vector[token] += 1

    return vector


def cosine_similarity(v1, v2):

    intersection = set(v1.keys()) & set(v2.keys())

    numerator = sum(v1[x] * v2[x] for x in intersection)

    sum1 = sum(v**2 for v in v1.values())
    sum2 = sum(v**2 for v in v2.values())

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if denominator == 0:
        return 0

    return numerator / denominator

# ============================================================
# SEGMENTATION DOCUMENT
# ============================================================

def split_document(text, chunk_size=120):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        if len(chunk) > 80:
            chunks.append(chunk)

    return chunks

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

        id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    def all(self):

        rows = self.conn.execute(

            "SELECT source,text FROM memory"

        ).fetchall()

        return rows

# ============================================================
# RAISONNEMENT TRIGRAM
# ============================================================

class TrigramReasoner:

    def generate_trigrams(self, tokens):

        trigrams = []

        for i in range(len(tokens)-2):

            trigrams.append(
                (tokens[i], tokens[i+1], tokens[i+2])
            )

        return trigrams

    def reasoning(self, question, memories):

        tokens = tokenize(question)

        trigrams = self.generate_trigrams(tokens)

        links = []

        for mem in memories:

            for tri in trigrams:

                if tri[0] in mem and tri[1] in mem:

                    links.append((tri, mem))

        return links

# ============================================================
# ANALYSE QUESTION
# ============================================================

class QuestionAnalyzer:

    def analyze(self, question):

        tokens = tokenize(question)

        analysis = {

            "tokens": tokens,
            "nombre_tokens": len(tokens),
            "complexite": len(set(tokens)),
            "type_question": "information"

        }

        if "pourquoi" in tokens:

            analysis["type_question"] = "causale"

        if "comment" in tokens:

            analysis["type_question"] = "explication"

        return analysis

# ============================================================
# GENERATEUR REPONSE
# ============================================================

class AnswerGenerator:

    def generate(self, question, memories, reasoning, analysis):

        response = []

        response.append("ANALYSE DE LA QUESTION")
        response.append(json.dumps(analysis, indent=2))

        response.append("\nCONNAISSANCES TROUVÉES")

        for m in memories:

            response.append("- " + m)

        response.append("\nRAISONNEMENT")

        for tri, mem in reasoning:

            response.append(
                f"Trigram {tri} relié à : {mem[:120]}"
            )

        response.append("\nCONCLUSION")

        if memories:

            response.append(memories[0])

        else:

            response.append(
                "Aucune connaissance pertinente trouvée."
            )

        return "\n".join(response)

# ============================================================
# RAPPORT ANALYTIQUE
# ============================================================

def generate_memory_report(memory):

    rows = memory.all()

    sources = Counter()

    concepts = Counter()

    for source,text in rows:

        sources[source] += 1

        tokens = tokenize(text)

        for t in tokens:

            concepts[t] += 1

    return {

        "souvenirs_totaux": len(rows),

        "sources": dict(sources),

        "concepts_dominants": dict(
            concepts.most_common(20)
        )

    }

# ============================================================
# ANALYSE DOCUMENT
# ============================================================

def document_analysis(text):

    words = len(text.split())

    sentences = len(text.split("."))

    return {

        "mots": words,
        "phrases": sentences,
        "longueur": len(text),
        "concepts_estimes": words // 6

    }

# ============================================================
# ORACLE BRAIN
# ============================================================

class OracleBrain:

    def __init__(self):

        self.memory = VectorMemory()

        self.reasoner = TrigramReasoner()

        self.analyzer = QuestionAnalyzer()

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

    def learn_document(self, text):

        chunks = split_document(text)

        learned = 0

        for c in chunks:

            tokens = tokenize(c)

            if len(tokens) > 5:

                self.memory.add(c, "document")

                learned += 1

        return learned

    # ========================================================
    # QUESTION
    # ========================================================

    def ask(self, question):

        analysis = self.analyzer.analyze(question)

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
    # RAPPORT IA
    # ========================================================

    def report(self):

        return generate_memory_report(self.memory)


# ============================================================
# TEST LOCAL
# ============================================================

if __name__ == "__main__":

    oracle = OracleBrain()

    print("\n🧠 ORACLE V15 Ω COSMOS INITIALISÉ\n")

    while True:

        q = input("QUESTION : ")

        if q == "exit":

            break

        print("\n", oracle.ask(q))