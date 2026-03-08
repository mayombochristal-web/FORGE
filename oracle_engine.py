import os
import sqlite3
import hashlib
import datetime
import re
import base64
import requests
import json
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

MEMORY_FOLDER = "oracle_memory"

DB_FILES = {
    "characters": "characters.db",
    "syllables": "syllables.db",
    "words": "words.db",
    "sentences": "sentences.db",
    "paragraphs": "paragraphs.db",
    "contexts": "contexts.db",
    "documents": "documents.db",
    "concept_graph": "concept_graph.db"
}

GITHUB_REPO = os.getenv("ORACLE_GITHUB_REPO")
GITHUB_TOKEN = os.getenv("ORACLE_GITHUB_TOKEN")

# ============================================================
# UTILITIES
# ============================================================

def ensure_folder():
    if not os.path.exists(MEMORY_FOLDER):
        os.makedirs(MEMORY_FOLDER)

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def simple_embedding(text):
    h = hashlib.sha256(text.encode()).hexdigest()
    return [int(h[i:i+4],16)/65535 for i in range(0,64,4)]

def cosine_similarity(v1,v2):

    dot=sum(a*b for a,b in zip(v1,v2))
    n1=sum(a*a for a in v1)**0.5
    n2=sum(a*a for a in v2)**0.5

    if n1==0 or n2==0:
        return 0

    return dot/(n1*n2)

# ============================================================
# TEXT PARSER
# ============================================================

def split_paragraphs(text):
    return text.split("\n\n")

def split_sentences(text):
    return re.split(r'[.!?]\s+',text)

def split_words(text):
    return re.findall(r'\b\w+\b',text.lower())

def split_syllables(word):
    return re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]|$)',word.lower())

# ============================================================
# GITHUB BACKUP
# ============================================================

def push_file_to_github(path):

    if not GITHUB_REPO or not GITHUB_TOKEN:
        return

    with open(path,"rb") as f:
        content=f.read()

    encoded=base64.b64encode(content).decode()

    filename=os.path.basename(path)

    url=f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

    headers={"Authorization":f"token {GITHUB_TOKEN}"}

    data={
        "message":"oracle memory update",
        "content":encoded
    }

    requests.put(url,json=data,headers=headers)

# ============================================================
# ORACLE ENGINE
# ============================================================

class OracleEngine:

    def __init__(self):

        ensure_folder()

        self.dbs={}

        for k,f in DB_FILES.items():

            path=os.path.join(MEMORY_FOLDER,f)

            conn=sqlite3.connect(path,check_same_thread=False)

            self.dbs[k]=conn

        self.init_tables()

        self.migrate_v19()

        self.repair_databases()

        self.reindex_embeddings()

        self.rebuild_concept_graph()

# ============================================================
# TABLE STRUCTURE
# ============================================================

    def init_tables(self):

        self.dbs["documents"].execute("""
        CREATE TABLE IF NOT EXISTS documents(
        id TEXT PRIMARY KEY,
        name TEXT,
        timestamp TEXT
        )
        """)

        self.dbs["paragraphs"].execute("""
        CREATE TABLE IF NOT EXISTS paragraphs(
        id TEXT PRIMARY KEY,
        text TEXT,
        embedding TEXT,
        document_id TEXT
        )
        """)

        self.dbs["sentences"].execute("""
        CREATE TABLE IF NOT EXISTS sentences(
        id TEXT PRIMARY KEY,
        text TEXT,
        embedding TEXT,
        paragraph_id TEXT,
        document_id TEXT
        )
        """)

        self.dbs["words"].execute("""
        CREATE TABLE IF NOT EXISTS words(
        id TEXT PRIMARY KEY,
        word TEXT,
        frequency INTEGER,
        sentence_id TEXT
        )
        """)

        self.dbs["syllables"].execute("""
        CREATE TABLE IF NOT EXISTS syllables(
        id TEXT PRIMARY KEY,
        syllable TEXT,
        word_id TEXT
        )
        """)

        self.dbs["characters"].execute("""
        CREATE TABLE IF NOT EXISTS characters(
        id TEXT PRIMARY KEY,
        char TEXT,
        word_id TEXT
        )
        """)

        self.dbs["contexts"].execute("""
        CREATE TABLE IF NOT EXISTS contexts(
        id TEXT PRIMARY KEY,
        topic TEXT
        )
        """)

        self.dbs["concept_graph"].execute("""
        CREATE TABLE IF NOT EXISTS relations(
        id TEXT PRIMARY KEY,
        concept1 TEXT,
        concept2 TEXT,
        weight INTEGER
        )
        """)

        for db in self.dbs.values():
            db.commit()

# ============================================================
# MIGRATION V19
# ============================================================

    def migrate_v19(self):

        try:

            cols=self.dbs["sentences"].execute(
            "PRAGMA table_info(sentences)"
            ).fetchall()

            names=[c[1] for c in cols]

            if "paragraph_id" not in names:
                self.dbs["sentences"].execute(
                "ALTER TABLE sentences ADD COLUMN paragraph_id TEXT"
                )

            if "document_id" not in names:
                self.dbs["sentences"].execute(
                "ALTER TABLE sentences ADD COLUMN document_id TEXT"
                )

            self.dbs["sentences"].commit()

        except:
            pass

# ============================================================
# DB REPAIR
# ============================================================

    def repair_databases(self):

        for name,db in self.dbs.items():

            try:
                db.execute("SELECT 1")
            except:
                db.close()

                path=os.path.join(MEMORY_FOLDER,DB_FILES[name])

                os.remove(path)

                self.dbs[name]=sqlite3.connect(path,check_same_thread=False)

# ============================================================
# REINDEX EMBEDDINGS
# ============================================================

    def reindex_embeddings(self):

        rows=self.dbs["sentences"].execute(
        "SELECT id,text FROM sentences"
        ).fetchall()

        for r in rows:

            emb=simple_embedding(r[1])

            self.dbs["sentences"].execute(
            "UPDATE sentences SET embedding=? WHERE id=?",
            (json.dumps(emb),r[0])
            )

        self.dbs["sentences"].commit()

# ============================================================
# CONTEXT EXTRACTION
# ============================================================

    def extract_context(self,text):

        words=split_words(text)

        freq=defaultdict(int)

        for w in words:
            freq[w]+=1

        top=sorted(freq.items(),key=lambda x:x[1],reverse=True)[:3]

        return " ".join([t[0] for t in top])

# ============================================================
# CONCEPT GRAPH REBUILD
# ============================================================

    def rebuild_concept_graph(self):

        self.dbs["concept_graph"].execute("DELETE FROM relations")

        rows=self.dbs["sentences"].execute(
        "SELECT text FROM sentences"
        ).fetchall()

        for r in rows:

            words=split_words(r[0])

            for i in range(len(words)-1):

                c1=words[i]
                c2=words[i+1]

                rid=hash_text(c1+c2)

                self.dbs["concept_graph"].execute(
                "INSERT OR IGNORE INTO relations VALUES(?,?,?,?)",
                (rid,c1,c2,1)
                )

        self.dbs["concept_graph"].commit()

# ============================================================
# LEARN DOCUMENT
# ============================================================

    def learn_document(self,file):

        text=file.read().decode("utf8","ignore")

        doc_id=hash_text(text)

        self.dbs["documents"].execute(
        "INSERT OR IGNORE INTO documents VALUES(?,?,?)",
        (doc_id,file.name,str(datetime.datetime.now()))
        )

        context=self.extract_context(text)

        cid=hash_text(context)

        self.dbs["contexts"].execute(
        "INSERT OR IGNORE INTO contexts VALUES(?,?)",
        (cid,context)
        )

        paragraphs=split_paragraphs(text)

        count=0

        for p in paragraphs:

            pid=hash_text(p)

            emb=simple_embedding(p)

            self.dbs["paragraphs"].execute(
            "INSERT OR IGNORE INTO paragraphs VALUES(?,?,?,?)",
            (pid,p,json.dumps(emb),doc_id)
            )

            sentences=split_sentences(p)

            for s in sentences:

                sid=hash_text(s)

                semb=simple_embedding(s)

                self.dbs["sentences"].execute(
                "INSERT OR IGNORE INTO sentences VALUES(?,?,?,?,?)",
                (sid,s,json.dumps(semb),pid,doc_id)
                )

                count+=1

        for db in self.dbs.values():
            db.commit()

        self.rebuild_concept_graph()

        return count

# ============================================================
# SEARCH
# ============================================================

    def search_sentences(self,query):

        qemb=simple_embedding(query)

        results=[]

        rows=self.dbs["sentences"].execute(
        "SELECT text,embedding FROM sentences"
        ).fetchall()

        for r in rows:

            try:
                emb=json.loads(r[1])
            except:
                continue

            score=cosine_similarity(qemb,emb)

            results.append((score,r[0]))

        results.sort(reverse=True)

        return results[:5]

# ============================================================
# REASONING
# ============================================================

    def reason(self,question):

        sentences=self.search_sentences(question)

        answer="Réponse basée sur la mémoire ORACLE :\n\n"

        for s in sentences:
            answer+="- "+s[1]+"\n"

        return answer

# ============================================================
# STATS
# ============================================================

    def stats(self):

        docs=self.dbs["documents"].execute(
        "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        sentences=self.dbs["sentences"].execute(
        "SELECT COUNT(*) FROM sentences"
        ).fetchone()[0]

        words=self.dbs["words"].execute(
        "SELECT COUNT(*) FROM words"
        ).fetchone()[0]

        return {
        "documents":docs,
        "souvenirs":sentences,
        "mots":words
        }