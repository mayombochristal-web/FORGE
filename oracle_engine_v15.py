import sqlite3
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

BASE_FOLDER="oracle_memory"

CORE_DB=os.path.join(BASE_FOLDER,"oracle_core.db")
DOC_DB=os.path.join(BASE_FOLDER,"oracle_documents.db")
DICT_DB=os.path.join(BASE_FOLDER,"oracle_dictionary.db")

if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)

# =====================================================
# STOPWORDS
# =====================================================

STOP_WORDS=set([
"le","la","les","de","des","du","un","une","et","ou",
"a","à","en","dans","sur","pour","par","est","sont",
"que","qui","quoi","comment"
])

# =====================================================
# NETTOYAGE TEXTE
# =====================================================

def clean_words(text):

    words=text.lower().split()

    return [w for w in words if w not in STOP_WORDS]

# =====================================================
# EXTRACTION CONCEPTS
# =====================================================

def extract_concepts(text):

    words=clean_words(text)

    counts=Counter(words)

    concepts=[w for w,c in counts.most_common(5)]

    return concepts

# =====================================================
# VECTOR
# =====================================================

def text_vector(text):

    words=clean_words(text)

    counts=Counter(words)

    vec=[]

    for w in counts:

        h=int(hashlib.md5(w.encode()).hexdigest(),16)%1000
        vec.append(h*counts[w])

    return vec

# =====================================================
# COSINE
# =====================================================

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

# =====================================================
# INIT DATABASE
# =====================================================

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

init_db()

# =====================================================
# ORACLE ENGINE
# =====================================================

class OracleEngine:

    def __init__(self):

        self.core=sqlite3.connect(CORE_DB,check_same_thread=False)
        self.docs=sqlite3.connect(DOC_DB,check_same_thread=False)

    # =================================================
    # LEARN
    # =================================================

    def learn(self,text,source="user"):

        vec=text_vector(text)

        self.core.execute(
        "INSERT INTO memories(text,vector,source,timestamp) VALUES(?,?,?,?)",
        (text,str(vec),source,datetime.now().isoformat())
        )

        self.core.commit()

    def learn_document(self,text,source="document"):

        vec=text_vector(text)

        self.docs.execute(
        "INSERT INTO documents(text,vector,source,timestamp) VALUES(?,?,?,?)",
        (text,str(vec),source,datetime.now().isoformat())
        )

        self.docs.commit()

    # =================================================
    # RECHERCHE INTELLIGENTE
    # =================================================

    def search(self,question,conn,table):

        qvec=text_vector(question)

        qwords=set(clean_words(question))

        qconcepts=set(extract_concepts(question))

        rows=conn.execute(f"SELECT text,vector,source FROM {table}")

        best=[]

        for r in rows.fetchall():

            try:

                text=r[0]
                vec=eval(r[1])
                source=r[2]

                score_cos=cosine(qvec,vec)

                mem_words=set(clean_words(text))

                overlap=len(qwords & mem_words)

                mem_concepts=set(extract_concepts(text))

                concept_match=len(qconcepts & mem_concepts)

                score=(score_cos*0.5)+(overlap*0.3)+(concept_match*0.2)

                if score>0.1:

                    best.append((score,text,source))

            except:

                continue

        best.sort(reverse=True)

        return best[:7]

    # =================================================
    # FUSION CONNAISSANCE
    # =================================================

    def synthesize(self,memories):

        if not memories:
            return None

        texts=[m[1] for m in memories[:3]]

        combined=" ".join(texts)

        return combined[:500]

    # =================================================
    # RAISONNEMENT
    # =================================================

    def reason(self,question):

        mem=self.search(question,self.core,"memories")

        docs=self.search(question,self.docs,"documents")

        knowledge=mem+docs

        synthesis=self.synthesize(knowledge)

        response="ANALYSE DE LA QUESTION\n\n"

        response+=question+"\n\n"

        response+="CONNAISSANCES IDENTIFIÉES\n\n"

        for s,t,src in knowledge[:5]:

            response+=f"- ({src}) {t[:200]}\n\n"

        response+="RAISONNEMENT\n\n"

        if synthesis:

            response+=synthesis+"\n\n"

        else:

            response+="Aucune connaissance suffisante pour répondre.\n\n"

        response+="CONCLUSION\n\n"

        response+="La réponse est construite à partir des connaissances apprises par l'ORACLE."

        return response

    # =================================================
    # STATS
    # =================================================

    def stats(self):

        try:

            n=self.core.execute(
            "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]

            return n

        except:

            return 0