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

    # =================================================
    # APPRENTISSAGE
    # =================================================

    def learn(self,text,source="user"):

        if not text:
            return

        vec=text_vector(text)

        try:

            self.conn.execute(
            "INSERT INTO memories(text,vector,source,timestamp) VALUES(?,?,?,?)",
            (text,str(vec),source,datetime.now().isoformat())
            )

            self.conn.commit()

            self.learn_trigram(text)

        except Exception as e:

            print("Learn error:",e)

    # =================================================
    # TRIGRAM LANGUAGE
    # =================================================

    def learn_trigram(self,text):

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

    # =================================================
    # RECHERCHE SEMANTIQUE
    # =================================================

    def search_memory(self,question):

        qvec=text_vector(question)

        cur=self.conn.execute("SELECT text,vector,source FROM memories")

        best=[]
        
        for row in cur.fetchall():

            mem=row[0]
            vec=eval(row[1])
            source=row[2]

            score=cosine(qvec,vec)

            if score>0.2:

                best.append((score,mem,source))

        best.sort(reverse=True)

        return best[:5]

    # =================================================
    # GENERATION
    # =================================================

    def generate(self,question):

        words=question.lower().split()

        if len(words)<2:
            return question

        w1,w2=words[-2],words[-1]

        cur=self.conn.execute(
        "SELECT w3,count FROM trigrams WHERE w1=? AND w2=? ORDER BY count DESC LIMIT 3",
        (w1,w2)
        )

        r=cur.fetchall()

        if not r:
            return question

        return question+" "+r[0][0]

    # =================================================
    # RAISONNEMENT ARGUMENTATIF
    # =================================================

    def reason(self,question):

        memories=self.search_memory(question)

        generated=self.generate(question)

        response="Analyse de la question : "+question+"\n\n"

        response+="Informations pertinentes trouvées dans la mémoire :\n\n"

        if memories:

            for s,m,src in memories:

                response+=f"- Source ({src}) : {m[:300]}\n\n"

        else:

            response+="Aucune information pertinente trouvée.\n\n"

        response+="Interprétation et réponse :\n\n"

        response+=generated+"\n\n"

        response+="Conclusion :\n"

        response+="La réponse est basée sur l'analyse des documents appris et sur la mémoire interne du système."

        return response

    # =================================================
    # STATS
    # =================================================

    def stats(self):

        try:

            n=self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            return n

        except:

            return 0