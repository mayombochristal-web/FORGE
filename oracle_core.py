# =====================================================
# 🧠 ORACLE CORE V6 — CERVEAU AUTONOME
# Architecture TTU-MC³ Stable
# =====================================================

import os, json, random, math, time, threading
from collections import Counter

# =====================================================
# CONFIG
# =====================================================

MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({}, open(LEXICON_PATH,"w",encoding="utf-8"))

# =====================================================
# ÉTAT GLOBAL DU CERVEAU
# =====================================================

brain_state = {
    "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
    "green_state":0.0,
    "hippocampus":[],
    "ghost_cache":{},
    "last_sleep":time.time()
}

lock = threading.Lock()

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev + 0.08*random.uniform(-1,1)

def consolidation_gate():
    brain_state["green_state"]=green_noise(
        brain_state["green_state"]
    )
    return abs(brain_state["green_state"])<0.25

# =====================================================
# Φ ENGINE
# =====================================================

def evolve_phi(exc):

    phi=brain_state["phi"]

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    s=sum(phi.values())
    for k in phi:
        phi[k]/=s

# =====================================================
# MÉMOIRE INCRÉMENTALE
# =====================================================

def load_lex():
    with lock:
        try:
            return json.load(open(LEXICON_PATH,"r",encoding="utf-8"))
        except:
            return {}

def save_lex(L):

    if len(L)>15000:
        L=dict(list(L.items())[-12000:])

    with lock:
        json.dump(
            L,
            open(LEXICON_PATH,"w",encoding="utf-8"),
            indent=2,
            ensure_ascii=False
        )

# =====================================================
# HIPPOCAMPUS
# =====================================================

def learn(text,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    phi=brain_state["phi"]
    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    brain_state["hippocampus"].append((words,energy))

    if len(brain_state["hippocampus"])>5 and consolidation_gate():
        consolidate()

def consolidate():

    L=load_lex()

    for words,energy in brain_state["hippocampus"]:
        for a,b in zip(words,words[1:]):
            L.setdefault(a,{})
            L[a][b]=L[a].get(b,0)+energy

    save_lex(L)
    brain_state["hippocampus"].clear()

# =====================================================
# SOMMEIL
# =====================================================

def sleep_cycle():

    L=load_lex()
    new={}

    for w,con in L.items():
        filt={t:v*0.997 for t,v in con.items() if v>1.2}
        if filt:
            new[w]=filt

    save_lex(new)
    brain_state["last_sleep"]=time.time()

def auto_sleep_loop():

    while True:
        time.sleep(30)

        if time.time()-brain_state["last_sleep"]>180:
            if consolidation_gate():
                sleep_cycle()

threading.Thread(target=auto_sleep_loop,
                 daemon=True).start()

# =====================================================
# GHOST CORTEX
# =====================================================

def ghost_preload(text):

    L=load_lex()
    cache={}

    for w in text.lower().split():
        if w in L:
            cache[w]=sorted(
                L[w].items(),
                key=lambda x:-x[1]
            )[:5]

    brain_state["ghost_cache"]=cache

# =====================================================
# CORTEX
# =====================================================

def contextual_seed(dialog):

    L=load_lex()
    ctx=" ".join(dialog).split()
    valid=[w for w in ctx if w in L]

    if valid:
        return Counter(valid).most_common(1)[0][0]

    return random.choice(list(L.keys())) if L else "oracle"

def associative(word):

    L=load_lex()

    ghost=brain_state["ghost_cache"].get(word)
    if ghost:
        return random.choice(ghost)[0]

    if word not in L:
        return word

    opts=L[word]
    phi=brain_state["phi"]

    if random.random()<phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

# =====================================================
# GÉNÉRATION
# =====================================================

def generate(dialog):

    L=load_lex()
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(dialog)
    words=[seed]
    used=set(words)

    phi=brain_state["phi"]
    length=int(10+phi["phi_m"]*30)

    for _ in range(length):
        nxt=associative(words[-1])

        if nxt in used and random.random()<phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    if phi["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."