import sqlite3
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

DB_FOLDER="oracle_memory"
DB_PATH=os.path.join(DB_FOLDER,"oracle.db")

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

def init_db():

    conn=sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    text TEXT,
    vector TEXT,
    source TEXT,
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

def text_vector(text):

    words=text.lower().split()

    counts=Counter(words)

    vector=[]

    for w in counts:

        h=int(hashlib.md5(w.encode()).hexdigest(),16)%1000

        vector.append(h*counts[w])

    return vector


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


class OracleEngine:

    def __init__(self):

        self.conn=sqlite3.connect(DB_PATH,check_same_thread=False)

    def learn(self,text,source="user"):

        vec=text_vector(text)

        self.conn.execute(
        "INSERT INTO memories(text,vector,source,timestamp) VALUES(?,?,?,?)",
        (text,str(vec),source,datetime.now().isoformat())
        )

        self.conn.commit()

        self.learn_trigram(text)

    def learn_trigram(self,text):

        w=text.lower().split()

        for i in range(len(w)-2):

            w1,w2,w3=w[i],w[i+1],w[i+2]

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

    def search(self,text):

        vec=text_vector(text)

        cur=self.conn.execute("SELECT text,vector FROM memories")

        best=""
        best_score=0

        for r in cur.fetchall():

            v=eval(r[1])

            s=cosine(vec,v)

            if s>best_score:

                best_score=s
                best=r[0]

        return best,best_score

    def generate(self,text):

        w=text.lower().split()

        if len(w)<2:
            return text

        w1,w2=w[-2],w[-1]

        cur=self.conn.execute(
        "SELECT w3,count FROM trigrams WHERE w1=? AND w2=? ORDER BY count DESC LIMIT 5",
        (w1,w2)
        )

        r=cur.fetchall()

        if not r:
            return text

        return text+" "+r[0][0]

    def reason(self,text):

        mem,score=self.search(text)

        gen=self.generate(text)

        answer=f"""
Analyse ORACLE

Question : {text}

Hypothèse générée :
{gen}

Mémoire associée (score {round(score,3)}) :

{mem}

Conclusion :

La réponse est basée sur la mémoire interne et l'analyse statistique des concepts.
"""

        return answer

    def stats(self):

        n=self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        return n