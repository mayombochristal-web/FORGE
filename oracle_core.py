# =====================================================
# 🧠 ORACLE CORE — COGNITIVE ENGINE V3.2
# =====================================================

import random, json, os, math, time
from collections import Counter

MEM_DIR="oracle_memory"
LEXICON_PATH=os.path.join(MEM_DIR,"lexicon.json")

os.makedirs(MEM_DIR,exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({},open(LEXICON_PATH,"w",encoding="utf-8"))

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    alpha=0.92
    return alpha*prev+(1-alpha)*random.uniform(-1,1)

def consolidation_gate(state):
    state["green_state"]=green_noise(state["green_state"])
    return abs(state["green_state"])<0.25

# =====================================================
# MEMORY
# =====================================================

def load_lex():
    try:
        return json.load(open(LEXICON_PATH,"r",encoding="utf-8"))
    except:
        return {}

def save_lex(L):
    json.dump(L,
        open(LEXICON_PATH,"w",encoding="utf-8"),
        indent=2,ensure_ascii=False)

# =====================================================
# Φ ENGINE
# =====================================================

def evolve_phi(state,exc):

    phi=state["phi"]

    noise=state["green_state"]
    phi["phi_c"]+=noise*0.05

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    total=sum(phi.values())
    for k in phi:
        phi[k]/=total

    state["identity_entropy"]+=random.uniform(-0.01,0.01)
    state["identity_entropy"]=max(
        0,min(1,state["identity_entropy"])
    )

# =====================================================
# LEARNING (HIPPOCAMPUS)
# =====================================================

def learn(state,text,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_lex()

    phi=state["phi"]
    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):
        L.setdefault(a,{})
        L[a][b]=L[a].get(b,0)+energy

    if consolidation_gate(state):
        save_lex(L)

# =====================================================
# PROCESS INPUT
# =====================================================

def process_input(state,text):

    ghost_factor=len(state["ghost_cache"])/5
    exc=min(1,(len(text)/200)+ghost_factor*0.2)

    evolve_phi(state,exc)
    learn(state,text,1.1)

    state["ghost_cache"].append(text)
    if len(state["ghost_cache"])>10:
        state["ghost_cache"].pop(0)

# =====================================================
# GENERATION
# =====================================================

def oracle_reply(state):

    L=load_lex()
    if not L:
        return "Mémoire vide."

    phi=state["phi"]

    seed=random.choice(list(L.keys()))
    words=[seed]

    length=int(8+phi["phi_m"]*35)

    for _ in range(length):

        if seed not in L:
            break

        opts=L[seed]
        nxt=random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

        words.append(nxt)
        seed=nxt

    if phi["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."

# =====================================================
# SLEEP
# =====================================================

def sleep_cycle(state):

    L=load_lex()
    new_L={}

    for w,con in L.items():
        filtered={t:v*0.997 for t,v in con.items() if v>1.2}
        if filtered:
            new_L[w]=filtered

    save_lex(new_L)
    state["last_sleep"]=time.time()

    return len(L)-len(new_L)

def auto_sleep(state):

    mem=load_lex()
    if len(mem)>800:
        sleep_cycle(state)