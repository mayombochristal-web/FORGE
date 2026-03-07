# ============================================================
# ORACLE MEMORY MANAGER
# ============================================================

import sqlite3
import os
import json
from datetime import datetime
from github_memory import backup_memory

MEMORY_FOLDER="oracle_memory"
DB=os.path.join(MEMORY_FOLDER,"cosmos_memory.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

class MemoryManager:

    def __init__(self):

        self.conn=sqlite3.connect(DB,check_same_thread=False)

        self.create()

    def create(self):

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

    # -------------------------------------------------

    def store(self,text,vector,source):

        self.conn.execute(

        "INSERT INTO memory(text,vector,source,date) VALUES(?,?,?,?)",

        (text,json.dumps(vector),source,str(datetime.now()))

        )

        self.conn.commit()

    # -------------------------------------------------

    def search(self,qvec):

        rows=self.conn.execute(

        "SELECT text,vector FROM memory"

        ).fetchall()

        import json,math

        results=[]

        for text,vjson in rows:

            vec=json.loads(vjson)

            inter=set(vec)&set(qvec)

            num=sum(vec[x]*qvec[x] for x in inter)

            s1=sum(v**2 for v in vec.values())
            s2=sum(v**2 for v in qvec.values())

            denom=math.sqrt(s1)*math.sqrt(s2)

            score=0

            if denom!=0:
                score=num/denom

            results.append((score,text))

        results.sort(reverse=True)

        return results[:5]

    # -------------------------------------------------

    def count(self):

        c=self.conn.execute("SELECT COUNT(*) FROM memory")

        return c.fetchone()[0]

    # -------------------------------------------------

    def all(self):

        rows=self.conn.execute(

        "SELECT text,source FROM memory"

        ).fetchall()

        data=[]

        for t,s in rows:

            data.append({"text":t,"source":s})

        return data