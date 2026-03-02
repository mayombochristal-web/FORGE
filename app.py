# =====================================================
# 🧠 ORACLE V3.1 — MODE SOMMEIL + GREEN NOISE
# Extension biologique TTU-MC³
# =====================================================

import streamlit as st
import random, json, os, math, time
import PyPDF2, docx
import pandas as pd
import speech_recognition as sr
from collections import deque, Counter

# =====================================================
# 1. CONFIGURATION
# =====================================================

MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({}, open(LEXICON_PATH, "w", encoding="utf-8"))

# =====================================================
# 2. SESSION STATE
# =====================================================

if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=40)

if "green_state" not in st.session_state:
    st.session_state.green_state = 0.0

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep = time.time()

# =====================================================
# 🧠 3. GREEN NOISE (HOMEOSTASIS)
# =====================================================

def green_noise(prev):
    alpha = 0.92
    return alpha*prev + (1-alpha)*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state = green_noise(
        st.session_state.green_state
    )
    return abs(st.session_state.green_state) < 0.25

# =====================================================
# 4. Φ ENGINE
# =====================================================

def evolve_phi(phi, excitation):

    phi["phi_m"] = min(1,max(0.1,phi["phi_m"]+excitation*0.15-0.01))
    phi["phi_c"] = min(1,max(0.1,phi["phi_c"]+excitation*0.3-0.03))
    phi["phi_d"] = min(1,max(0.1,phi["phi_d"]+0.02-excitation*0.05))

    total=sum(phi.values())
    for k in phi:
        phi[k]/=total

    return phi

# =====================================================
# 5. MÉMOIRE
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
# 6. APPRENTISSAGE SYNAPTIQUE
# =====================================================

def learn(text, phi, importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_lex()

    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):
        L.setdefault(a,{})
        L[a][b]=L[a].get(b,0)+energy

    # écriture seulement si cerveau stable
    if consolidation_gate():
        save_lex(L)

# =====================================================
# 🧠 7. MODE SOMMEIL (CONSOLIDATION)
# =====================================================

def sleep_cycle():

    L=load_lex()
    new_L={}

    for w,con in L.items():

        filtered={
            t:v*0.997   # décroissance biologique lente
            for t,v in con.items()
            if v>1.2
        }

        if filtered:
            new_L[w]=filtered

    save_lex(new_L)

    st.session_state.last_sleep=time.time()

    return f"🌙 Sommeil terminé — {len(L)-len(new_L)} synapses oubliées"

# =====================================================
# 8. THALAMUS
# =====================================================

def contextual_seed(L):

    context=" ".join(st.session_state.dialog_memory).split()
    c=[w for w in context if w in L]

    if c:
        return Counter(c).most_common(1)[0][0]

    return random.choice(list(L.keys()))

# =====================================================
# 9. CORTEX
# =====================================================

def logical_layer(seq,L):
    return [w for w in seq if w in L] or seq

def associative_layer(word,L,phi):

    if word not in L:
        return word

    opts=L[word]

    if random.random()<phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

def predictive_layer(words,phi):

    if phi["phi_m"]>0.6:
        words.append("continuité")

    if phi["phi_c"]>0.65:
        words.append("évolution")

    return words

# =====================================================
# 10. GÉNÉRATION
# =====================================================

def oracle_reply(phi):

    L=load_lex()
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(L)

    words=[seed]
    used=set(words)

    length=int(8+phi["phi_m"]*35)

    for _ in range(length):

        filtered=logical_layer(words,L)
        current=filtered[-1]

        nxt=associative_layer(current,L,phi)

        if nxt in used and random.random()<phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    words=predictive_layer(words,phi)

    return " ".join(words).capitalize()+"."

# =====================================================
# 🧠 AUTO SOMMEIL (BIOLOGIQUE)
# =====================================================

def auto_sleep():

    elapsed=time.time()-st.session_state.last_sleep

    # toutes les ~3 minutes cognitives
    if elapsed>180 and consolidation_gate():
        sleep_cycle()

auto_sleep()

# =====================================================
# 11. UI
# =====================================================

st.set_page_config(page_title="ORACLE V3.1",page_icon="🧠")

st.title("🧠 ORACLE V3.1 — Cognitive Sleep Engine")

user_msg=st.text_input("Parlez à l'Oracle")

if st.button("➡️") and user_msg:

    st.session_state.dialog_memory.append(user_msg)

    excitation=min(1,len(user_msg)/200)

    st.session_state.phi=evolve_phi(
        st.session_state.phi,excitation
    )

    learn(user_msg,st.session_state.phi,1.1)

    reply=oracle_reply(st.session_state.phi)

    st.session_state.dialog_memory.append(reply)

    learn(reply,{"phi_m":0.1,"phi_c":0.1,"phi_d":0.1},0.3)

for msg in st.session_state.dialog_memory:
    st.write(msg)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    size=os.path.getsize(LEXICON_PATH)/1024
    st.success(f"Mémoire : {size:.2f} KB")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}: {v:.2f}")

    st.divider()

    if st.button("🌙 Sommeil forcé"):
        st.warning(sleep_cycle())
        st.rerun()