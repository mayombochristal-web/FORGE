import sqlite3
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

# =========================================================
# ORACLE V15 Ω COSMOS ENGINE
# =========================================================

BASE_FOLDER="oracle_memory"

CORE_DB=os.path.join(BASE_FOLDER,"oracle_core.db")
DOC_DB=os.path.join(BASE_FOLDER,"oracle_documents.db")
DICT_DB=os.path.join(BASE_FOLDER,"oracle_dictionary.db")

if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)


# =========================================================
# DATABASE INIT
# =========================================================

def init_db():

    conn=sqlite3.connect(CORE_DB)
    c=conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    vector TEXT,
    source TEXT,
    timestamp TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS trigrams(
    w1 TEXT,
    w2 TEXT,
    w3 TEXT,
    count INTEGER
    )
    """)

    conn.commit()
    conn.close()


    conn=sqlite3.connect(DOC_DB)
    c=conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    vector TEXT,
    source TEXT,
    timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


    conn=sqlite3.connect(DICT_DB)
    c=conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS dictionary(
    word TEXT,
    definition TEXT
    )
    """)

    conn.commit()
    conn.close()


init_db()

# =========================================================
# VECTOR
# =========================================================

def text_vector(text):

    words=text.lower().split()
    counts=Counter(words)

    vec=[]

    for w in counts:

        h=int(hashlib.md5(w.encode()).hexdigest(),16)%1000
        vec.append(h*counts[w])

    return vec


def cosine(a,b):

    if not a or not b:
        return 0

    n=min(len(a),len(b))

    dot=sum(a[i]*b[i] for i in range(n))

    na=math.sqrt(sum(x*x for x in a))
    nb=math.sqrt(sum(x*x for x in b))

    if na==0 or nb==0:
        return 0

    return dot/(na*nb)

# =========================================================
# ORACLE ENGINE
# =========================================================

class OracleEngine:

    def __init__(self):

        self.core=sqlite3.connect(CORE_DB,check_same_thread=False)
        self.docs=sqlite3.connect(DOC_DB,check_same_thread=False)
        self.dict=sqlite3.connect(DICT_DB,check_same_thread=False)

    # =====================================================
    # LEARN
    # =====================================================

    def learn(self,text,source="user"):

        vec=text_vector(text)

        self.core.execute(
        "INSERT INTO memories(text,vector,source,timestamp) VALUES(?,?,?,?)",
        (text,str(vec),source,datetime.now().isoformat())
        )

        self.core.commit()

        self.learn_trigram(text)

    def learn_document(self,text,source="document"):

        vec=text_vector(text)

        self.docs.execute(
        "INSERT INTO documents(text,vector,source,timestamp) VALUES(?,?,?,?)",
        (text,str(vec),source,datetime.now().isoformat())
        )

        self.docs.commit()

    # =====================================================
    # TRIGRAM
    # =====================================================

    def learn_trigram(self,text):

        words=text.lower().split()

        for i in range(len(words)-2):

            w1,w2,w3=words[i],words[i+1],words[i+2]

            cur=self.core.execute(
            "SELECT count FROM trigrams WHERE w1=? AND w2=? AND w3=?",
            (w1,w2,w3)
            ).fetchone()

            if cur:

                self.core.execute(
                "UPDATE trigrams SET count=count+1 WHERE w1=? AND w2=? AND w3=?",
                (w1,w2,w3)
                )

            else:

                self.core.execute(
                "INSERT INTO trigrams VALUES(?,?,?,1)",
                (w1,w2,w3)
                )

        self.core.commit()

    # =====================================================
    # SEARCH
    # =====================================================

    def search(self,question,conn,table):

        qvec=text_vector(question)

        rows=conn.execute(f"SELECT text,vector,source FROM {table}")

        best=[]

        for r in rows.fetchall():

            vec=eval(r[1])

            score=cosine(qvec,vec)

            if score>0.2:

                best.append((score,r[0],r[2]))

        best.sort(reverse=True)

        return best[:5]

    # =====================================================
    # AUTO SOURCE
    # =====================================================

    def select_sources(self,question):

        q=question.lower()

        sources=["memories"]

        if "document" in q or "analyse" in q:
            sources.append("documents")

        if "signifie" in q or "definition" in q:
            sources.append("dictionary")

        return sources

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(self,question):

        words=question.lower().split()

        if len(words)<2:
            return question

        w1,w2=words[-2],words[-1]

        cur=self.core.execute(
        "SELECT w3,count FROM trigrams WHERE w1=? AND w2=? ORDER BY count DESC LIMIT 1",
        (w1,w2)
        ).fetchone()

        if not cur:
            return question

        return question+" "+cur[0]

    # =====================================================
    # REASON
    # =====================================================

    def reason(self,question):

        sources=self.select_sources(question)

        memories=self.search(question,self.core,"memories")

        docs=[]

        if "documents" in sources:

            docs=self.search(question,self.docs,"documents")

        generated=self.generate(question)

        response="ANALYSE DE LA QUESTION\n\n"
        response+=question+"\n\n"

        response+="CONNAISSANCES TROUVÉES\n\n"

        for s,m,src in memories:

            response+=f"- ({src}) {m[:200]}\n\n"

        for s,m,src in docs:

            response+=f"- (document) {m[:200]}\n\n"

        response+="RAISONNEMENT\n\n"

        response+=generated+"\n\n"

        response+="CONCLUSION\n\n"

        response+="Cette réponse est basée sur les connaissances stockées dans les bases mémoire de l'ORACLE."

        return response

    # =====================================================
    # STATS
    # =====================================================

    def stats(self):

        n=self.core.execute(
        "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        return n