# ============================================================
# ORACLE V14 Ω ULTRA STABLE
# TTU Cognitive Engine
# Vector Memory + Multi Concept Reasoning
# ============================================================

import sqlite3
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DB_FOLDER = "oracle_memory"
DB_PATH = os.path.join(DB_FOLDER, "oracle.db")

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# ============================================================
# DATABASE INITIALISATION
# ============================================================

def init_db():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        vector TEXT,
        timestamp TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS trigrams(
        w1 TEXT,
        w2 TEXT,
        w3 TEXT,
        count INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ============================================================
# VECTOR EMBEDDING SIMPLIFIÉ
# ============================================================

def text_to_vector(text):

    words = text.lower().split()
    counts = Counter(words)

    vector = []

    for w in counts:
        h = int(hashlib.md5(w.encode()).hexdigest(),16)%1000
        vector.append(h * counts[w])

    return vector

def cosine_similarity(a,b):

    if not a or not b:
        return 0

    min_len=min(len(a),len(b))

    dot=sum(a[i]*b[i] for i in range(min_len))
    normA=math.sqrt(sum(x*x for x in a))
    normB=math.sqrt(sum(x*x for x in b))

    if normA==0 or normB==0:
        return 0

    return dot/(normA*normB)

# ============================================================
# ORACLE ENGINE
# ============================================================

class OracleEngine:

    def __init__(self):

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # ========================================================
    # MEMORY
    # ========================================================

    def store(self,text):

        vector=text_to_vector(text)

        self.conn.execute(
            "INSERT INTO memories(text,vector,timestamp) VALUES(?,?,?)",
            (text,str(vector),datetime.now().isoformat())
        )

        self.conn.commit()

        self.learn_trigrams(text)

    # ========================================================
    # TRIGRAM LEARNING
    # ========================================================

    def learn_trigrams(self,text):

        words=text.lower().split()

        for i in range(len(words)-2):

            w1,w2,w3=words[i],words[i+1],words[i+2]

            cur=self.conn.execute(
                "SELECT count FROM trigrams WHERE w1=? AND w2=? AND w3=?",
                (w1,w2,w3)
            ).fetchone()

            if cur:

                self.conn.execute(
                    "UPDATE trigrams SET count=count+1 WHERE w1=? AND w2=? AND w3=?",
                    (w1,w2,w3)
                )

            else:

                self.conn.execute(
                    "INSERT INTO trigrams VALUES(?,?,?,1)",
                    (w1,w2,w3)
                )

        self.conn.commit()

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def search_memory(self,text):

        vector=text_to_vector(text)

        cur=self.conn.execute("SELECT text,vector FROM memories")

        best_score=0
        best_text=""

        for row in cur.fetchall():

            mem=row[0]
            vec=eval(row[1])

            score=cosine_similarity(vector,vec)

            if score>best_score:

                best_score=score
                best_text=mem

        return best_text

    # ========================================================
    # TRIGRAM GENERATION
    # ========================================================

    def generate(self,seed):

        words=seed.lower().split()

        if len(words)<2:
            return seed

        w1,w2=words[-2],words[-1]

        cur=self.conn.execute(
            "SELECT w3,count FROM trigrams WHERE w1=? AND w2=? ORDER BY count DESC LIMIT 5",
            (w1,w2)
        )

        options=cur.fetchall()

        if not options:
            return seed

        next_word=options[0][0]

        return seed+" "+next_word

    # ========================================================
    # MULTI CONCEPT REASONING
    # ========================================================

    def reason(self,text):

        memory=self.search_memory(text)

        generated=self.generate(text)

        if memory:

            return f"{generated}\n\nConcept associé : {memory}"

        return generated

    # ========================================================
    # STATS
    # ========================================================

    def memory_size(self):

        try:

            n=self.conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]

            return n

        except:

            return 0