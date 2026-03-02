# =====================================================
# 🧠 ORACLE V6 CORE — CORTEX AUTONOME
# Compatible V3.1 Biology
# =====================================================

import sqlite3, random, math, time
from collections import Counter

DB="memory.db"

# ================= MEMORY INIT =================

def init_db():
    conn=sqlite3.connect(DB)
    c=conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS synapses(
        w1 TEXT,
        w2 TEXT,
        weight REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= GREEN NOISE =================

green_state=0.0

def green_noise(prev):
    return 0.92*prev + 0.08*random.uniform(-1,1)

def consolidation_gate():
    global green_state
    green_state=green_noise(green_state)
    return abs(green_state)<0.25

# ================= Φ ENGINE =================

def evolve_phi(phi,exc):

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    s=sum(phi.values())
    for k in phi:
        phi[k]/=s

    return phi

# ================= LEARNING =================

def learn(text,phi):

    words=text.lower().split()
    if len(words)<2:
        return

    energy=math.sqrt(sum(v*v for v in phi.values()))

    conn=sqlite3.connect(DB)
    c=conn.cursor()

    for a,b in zip(words,words[1:]):
        c.execute("""
        INSERT INTO synapses VALUES(?,?,?)
        """,(a,b,energy))

    conn.commit()
    conn.close()

# ================= MEMORY QUERY =================

def next_word(word,phi):

    conn=sqlite3.connect(DB)
    c=conn.cursor()

    c.execute("""
    SELECT w2, SUM(weight)
    FROM synapses
    WHERE w1=?
    GROUP BY w2
    """,(word,))

    rows=c.fetchall()
    conn.close()

    if not rows:
        return word

    words,weights=zip(*rows)

    if random.random()<phi["phi_c"]:
        return random.choices(words,weights)[0]

    return words[weights.index(max(weights))]

# ================= GENERATION =================

def reply(dialog,phi):

    conn=sqlite3.connect(DB)
    c=conn.cursor()

    c.execute("SELECT w1 FROM synapses LIMIT 100")
    seeds=[r[0] for r in c.fetchall()]
    conn.close()

    if not seeds:
        return "Mémoire vide."

    context=" ".join(dialog).split()
    valid=[w for w in context if w in seeds]

    seed=random.choice(valid or seeds)

    words=[seed]
    used=set(words)

    for _ in range(int(10+phi["phi_m"]*30)):

        nxt=next_word(words[-1],phi)

        if nxt in used and random.random()<phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    if phi["phi_m"]>0.6:
        words.append("continuité")

    if phi["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."