import sqlite3
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

# =========================================================
# ORACLE V15 Ω COSMOS+
# ENGINE AMELIORE
# =========================================================

BASE_FOLDER="oracle_memory"

CORE_DB=os.path.join(BASE_FOLDER,"oracle_core.db")
DOC_DB=os.path.join(BASE_FOLDER,"oracle_documents.db")
DICT_DB=os.path.join(BASE_FOLDER,"oracle_dictionary.db")

if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)

# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS=set([
"le","la","les","de","des","du",
"un","une","et","ou","a","à",
"en","dans","sur","pour","par",
"que","qui","quoi","est","sont"
])

# =========================================================
# INITIALISATION DB
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
# NETTOYAGE TEXTE
# =========================================================

def clean_words(text):

    words=text.lower().split()

    return [w for w in words if w not in STOP_WORDS]

# =========================================================
# VECTORISATION
# =========================================================

def text_vector(text):

    words=clean_words(text)

    counts=Counter(words)

    vec=[]

    for w in counts:

        h=int(hashlib.md5(w.encode()).hexdigest(),16)%1000
        vec.append(h*counts[w])

    return vec

# =========================================================
# COSINE
# =========================================================

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

        if not text:
            return

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

        words=clean_words(text)

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
    # RECHERCHE HYBRIDE
    # =====================================================

    def search(self,question,conn,table):

        qvec=text_vector(question)

        qwords=set(clean_words(question))

        rows=conn.execute(f"SELECT text,vector,source FROM {table}")

        best=[]

        for r in rows.fetchall():

            try:

                mem_text=r[0]

                vec=eval(r[1])

                source=r[2]

                # cosine similarity
                score=cosine(qvec,vec)

                # lexical overlap
                mem_words=set(clean_words(mem_text))

                overlap=len(qwords & mem_words)

                score=score+(overlap*0.1)

                if score>0.05:

                    best.append((score,mem_text,source))

            except:

                continue

        best.sort(reverse=True)

        return best[:5]

    # =====================================================
    # GENERATION TRIGRAM
    # =====================================================

    def generate(self,question):

        words=clean_words(question)

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
    # EXTRACTION CONNAISSANCE
    # =====================================================

    def extract_answer(self,memories):

        if not memories:
            return None

        best=memories[0][1]

        return best

    # =====================================================
    # RAISONNEMENT
    # =====================================================

    def reason(self,question):

        memories=self.search(question,self.core,"memories")

        docs=self.search(question,self.docs,"documents")

        generated=self.generate(question)

        best_answer=self.extract_answer(memories)

        response="ANALYSE DE LA QUESTION\n\n"

        response+=question+"\n\n"

        response+="CONNAISSANCES TROUVÉES\n\n"

        if memories:

            for s,m,src in memories:

                response+=f"- ({src}) {m[:250]}\n\n"

        if docs:

            for s,m,src in docs:

                response+=f"- (document) {m[:250]}\n\n"

        if not memories and not docs:

            response+="Aucune connaissance pertinente trouvée.\n\n"

        response+="RAISONNEMENT\n\n"

        if best_answer:

            response+="Les informations retrouvées dans la mémoire indiquent que :\n\n"

            response+=best_answer+"\n\n"

        else:

            response+=generated+"\n\n"

        response+="CONCLUSION\n\n"

        response+="La réponse est produite à partir des connaissances stockées dans les bases mémoire et de l'analyse contextuelle de la question."

        return response

    # =====================================================
    # STATS
    # =====================================================

    def stats(self):

        try:

            n=self.core.execute(
            "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]

            return n

        except:

            return 0