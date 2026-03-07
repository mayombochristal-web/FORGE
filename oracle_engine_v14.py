# =====================================================
# ORACLE V14 Ω ENGINE
# =====================================================

import sqlite3
import numpy as np
import pandas as pd
import math
import os
import json
import io
import random
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px

try:
    import PyPDF2
    PDF_AVAILABLE=True
except:
    PDF_AVAILABLE=False

try:
    import docx
    DOCX_AVAILABLE=True
except:
    DOCX_AVAILABLE=False

MEM="oracle_memory"
os.makedirs(MEM,exist_ok=True)

DB=os.path.join(MEM,"relations.db")

model=SentenceTransformer("all-MiniLM-L6-v2")

TARGET={"m":0.62,"c":0.71,"d":0.88}

# =====================================================
# INITIALISATION DB
# =====================================================

def init_db():

    conn=sqlite3.connect(DB)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS trigrams(
    w1 TEXT,
    w2 TEXT,
    w3 TEXT,
    weight REAL,
    PRIMARY KEY(w1,w2,w3))
    """)

    conn.commit()
    conn.close()

init_db()

# =====================================================
# ORACLE BRAIN
# =====================================================

class OracleBrain:

    def __init__(self):

        self.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

        self.age=0

    # =================================================
    # TTU DISTANCE
    # =================================================

    def distance(self):

        p=self.phi

        return math.sqrt(
        (p["phi_m"]-TARGET["m"])**2+
        (p["phi_c"]-TARGET["c"])**2+
        (p["phi_d"]-TARGET["d"])**2)

    # =================================================
    # EVOLUTION TTU
    # =================================================

    def evolve(self,exc):

        p=self.phi

        err=self.distance()

        m=p["phi_m"]+exc*0.1-(p["phi_m"]-TARGET["m"])*0.2
        c=p["phi_c"]+exc*0.2-(p["phi_c"]-TARGET["c"])*0.2
        d=p["phi_d"]+0.04+err*0.15

        norm=math.sqrt(m*m+c*c+d*d)

        self.phi={
        "phi_m":m/norm,
        "phi_c":c/norm,
        "phi_d":d/norm
        }

    # =================================================
    # TOKENIZE
    # =================================================

    def tokenize(self,text):

        return [w.lower() for w in text.split() if len(w)>1]

    # =================================================
    # LEARN
    # =================================================

    def learn(self,text):

        words=self.tokenize(text)

        if len(words)<3:
            return

        conn=sqlite3.connect(DB)

        for i in range(len(words)-2):

            a,b,d=words[i],words[i+1],words[i+2]

            conn.execute("""
            INSERT INTO trigrams VALUES(?,?,?,1)
            ON CONFLICT(w1,w2,w3)
            DO UPDATE SET weight=weight+1
            """,(a,b,d))

        conn.commit()
        conn.close()

        self.age+=len(words)

        self.evolve(0.5)

    # =================================================
    # VECTOR SEARCH
    # =================================================

    def semantic_search(self,text):

        emb=model.encode([text])[0]

        conn=sqlite3.connect(DB)

        words=pd.read_sql_query(
        "SELECT DISTINCT w1 FROM trigrams LIMIT 500",
        conn)

        conn.close()

        if len(words)==0:
            return []

        emb_words=model.encode(words["w1"].tolist())

        scores=np.dot(emb_words,emb)

        idx=np.argsort(scores)[-5:]

        return words.iloc[idx]["w1"].tolist()

    # =================================================
    # THINK
    # =================================================

    def think(self,text=""):

        concepts=self.semantic_search(text)

        conn=sqlite3.connect(DB)

        if concepts:

            seed=random.choice(concepts)

            q=conn.execute(
            "SELECT w2 FROM trigrams WHERE w1=? LIMIT 1",
            (seed,)
            ).fetchone()

            if q:
                words=[seed,q[0]]
            else:
                words=[seed]
        else:

            r=conn.execute(
            "SELECT w1,w2 FROM trigrams ORDER BY RANDOM() LIMIT 1"
            ).fetchone()

            if not r:
                return "Donne moi du texte."

            words=[r[0],r[1]]

        for _ in range(20):

            rows=conn.execute(
            "SELECT w3,weight FROM trigrams WHERE w1=? AND w2=?",
            (words[-2],words[-1])
            ).fetchall()

            if not rows:
                break

            ws=[r[0] for r in rows]

            w=np.array([r[1] for r in rows])

            p=w/w.sum()

            nxt=np.random.choice(ws,p=p)

            words.append(nxt)

        conn.close()

        return " ".join(words).capitalize()+"."

    # =================================================
    # DOCUMENT READER
    # =================================================

    def read_document(self,file):

        name=file.name.lower()

        try:

            if name.endswith(".txt"):
                return file.read().decode("utf-8","ignore")

            if name.endswith(".csv"):
                return pd.read_csv(file).to_string()

            if name.endswith(".json"):
                return json.dumps(json.load(file))

            if name.endswith(".xlsx"):
                return pd.read_excel(file).to_string()

            if name.endswith(".docx") and DOCX_AVAILABLE:
                doc=docx.Document(io.BytesIO(file.read()))
                return " ".join(p.text for p in doc.paragraphs)

            if name.endswith(".pdf") and PDF_AVAILABLE:
                reader=PyPDF2.PdfReader(io.BytesIO(file.read()))
                text=""
                for p in reader.pages:
                    text+=p.extract_text() or ""
                return text

        except:
            return ""

        return ""

    # =================================================
    # DOCUMENT ANALYSIS
    # =================================================

    def analyze_document(self,text):

        words=self.tokenize(text)

        self.learn(text)

        stats={
        "mots":len(words),
        "concepts":len(set(words)),
        "top":Counter(words).most_common(10)
        }

        return stats

    # =================================================
    # SLEEP
    # =================================================

    def sleep_cycle(self):

        conn=sqlite3.connect(DB)

        df=pd.read_sql_query("SELECT weight FROM trigrams",conn)

        if len(df)>5:

            th=df.weight.mean()-df.weight.std()

            conn.execute(
            "DELETE FROM trigrams WHERE weight<?",
            (th,)
            )

        conn.commit()
        conn.close()

    # =================================================
    # MEMORY SIZE
    # =================================================

    def memory_size(self):

        conn=sqlite3.connect(DB)

        n=conn.execute(
        "SELECT COUNT(*) FROM trigrams"
        ).fetchone()[0]

        conn.close()

        return n

    # =================================================
    # EXPORT
    # =================================================

    def export_concepts(self):

        conn=sqlite3.connect(DB)

        df=pd.read_sql_query(
        "SELECT * FROM trigrams",
        conn)

        conn.close()

        return df

    # =================================================
    # VISUALISATION
    # =================================================

    def visualize(self):

        conn=sqlite3.connect(DB)

        words=pd.read_sql_query(
        "SELECT DISTINCT w1 FROM trigrams LIMIT 200",
        conn)

        conn.close()

        if len(words)<5:
            return None

        emb=model.encode(words["w1"].tolist())

        tsne=TSNE(n_components=2)

        coords=tsne.fit_transform(emb)

        df=pd.DataFrame({
        "word":words["w1"],
        "x":coords[:,0],
        "y":coords[:,1]
        })

        fig=px.scatter(df,x="x",y="y",text="word")

        return fig