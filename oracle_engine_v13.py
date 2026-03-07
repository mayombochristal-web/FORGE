import sqlite3
import numpy as np
import random
import math
import os
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

MEM = "oracle_memory"
os.makedirs(MEM, exist_ok=True)

DB = os.path.join(MEM, "relations.db")

TARGET = {"m":0.62,"c":0.71,"d":0.88}

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# INITIALISATION DB
# =====================================================

def init_db():

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
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
# ORACLE ENGINE
# =====================================================

class OracleBrain:

    def __init__(self):

        self.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

        self.dialog=deque(maxlen=40)

        self.age=0

    # =================================================
    # DISTANCE ATTRACTEUR
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

        error=self.distance()

        m=p["phi_m"]+exc*0.1-(p["phi_m"]-TARGET["m"])*0.2
        c=p["phi_c"]+exc*0.2-(p["phi_c"]-TARGET["c"])*0.2
        d=p["phi_d"]+0.04+error*0.15

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
        c=conn.cursor()

        for i in range(len(words)-2):

            a,b,d=words[i],words[i+1],words[i+2]

            c.execute("""
            INSERT INTO trigrams VALUES(?,?,?,1)
            ON CONFLICT(w1,w2,w3)
            DO UPDATE SET weight=weight+1
            """,(a,b,d))

        conn.commit()
        conn.close()

        self.age+=len(words)

        self.evolve(0.6)

    # =================================================
    # NEXT WORD
    # =================================================

    def next(self,a,b):

        conn=sqlite3.connect(DB)
        c=conn.cursor()

        c.execute("SELECT w3,weight FROM trigrams WHERE w1=? AND w2=?",(a,b))

        rows=c.fetchall()

        conn.close()

        if not rows:
            return None

        words=[r[0] for r in rows]

        weights=np.array([r[1] for r in rows])

        p=weights/weights.sum()

        return np.random.choice(words,p=p)

    # =================================================
    # THINK
    # =================================================

    def think(self):

        conn=sqlite3.connect(DB)
        c=conn.cursor()

        c.execute("SELECT w1,w2 FROM trigrams ORDER BY RANDOM() LIMIT 1")

        seed=c.fetchone()

        conn.close()

        if not seed:
            return "Donne moi du texte."

        words=[seed[0],seed[1]]

        for _ in range(20):

            nxt=self.next(words[-2],words[-1])

            if not nxt:
                break

            words.append(nxt)

        self.dialog.append(" ".join(words))

        return " ".join(words).capitalize()+"."

    # =================================================
    # FEEDBACK
    # =================================================

    def feedback(self,text,positive=True):

        words=self.tokenize(text)

        delta=0.4 if positive else -0.2

        conn=sqlite3.connect(DB)
        c=conn.cursor()

        for i in range(len(words)-2):

            a,b,d=words[i],words[i+1],words[i+2]

            c.execute("""
            UPDATE trigrams
            SET weight=weight+?
            WHERE w1=? AND w2=? AND w3=?
            """,(delta,a,b,d))

        conn.commit()
        conn.close()

    # =================================================
    # SLEEP
    # =================================================

    def sleep_cycle(self):

        conn=sqlite3.connect(DB)

        df=pd.read_sql_query("SELECT weight FROM trigrams",conn)

        if len(df)<5:
            return

        threshold=df.weight.mean()-df.weight.std()

        conn.execute(
        "DELETE FROM trigrams WHERE weight<?",
        (threshold,)
        )

        conn.commit()
        conn.close()

    # =================================================
    # MEMORY SIZE
    # =================================================

    def memory_size(self):

        conn=sqlite3.connect(DB)

        c=conn.cursor()

        c.execute("SELECT COUNT(*) FROM trigrams")

        n=c.fetchone()[0]

        conn.close()

        return n

    # =================================================
    # VISUALISATION
    # =================================================

    def visualize(self):

        conn=sqlite3.connect(DB)

        words=pd.read_sql_query(
        "SELECT DISTINCT w1 FROM trigrams LIMIT 200",conn)

        conn.close()

        if len(words)<10:
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