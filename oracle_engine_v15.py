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
# TOKENIZER
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
# VECTOR
# ============================================================

def text_vector(tokens):

    vector = defaultdict(int)

    for t in tokens:
        vector[t] += 1

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
# DOCUMENT SPLIT
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
# MEMORY
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

            (text, json.dumps(vector), source, str(datetime.now()))
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
# ORACLE ENGINE
# ============================================================

class OracleEngine:

    def __init__(self):

        self.memory = VectorMemory()

    # --------------------------------------------------------

    def learn(self, text, source="user"):

        if len(text.strip()) > 10:

            self.memory.add(text, source)

    # --------------------------------------------------------

    def learn_document(self, text):

        chunks = split_document(text)

        learned = 0

        for c in chunks:

            tokens = tokenize(c)

            if len(tokens) > 5:

                self.memory.add(c, "document")

                learned += 1

        return learned

    # --------------------------------------------------------

    def reason(self, question):

        results = self.memory.search(question)

        memories = [r[1] for r in results]

        response = []

        response.append("ANALYSE DE LA QUESTION")
        response.append(question)

        response.append("\nCONNAISSANCES TROUVÉES")

        for m in memories:

            response.append("- " + m[:200])

        response.append("\nCONCLUSION")

        if memories:

            response.append(memories[0])

        else:

            response.append("Aucune connaissance trouvée.")

        return "\n".join(response)

    # --------------------------------------------------------

    def stats(self):

        rows = self.memory.all()

        return len(rows)

    # --------------------------------------------------------

    def report(self):

        rows = self.memory.all()

        sources = Counter()

        concepts = Counter()

        for source,text in rows:

            sources[source]+=1

            tokens = tokenize(text)

            for t in tokens:

                concepts[t]+=1

        return {

            "souvenirs_totaux": len(rows),

            "sources": dict(sources),

            "concepts": dict(concepts.most_common(20))

        }