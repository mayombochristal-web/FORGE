import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
import docx
import pandas as pd
import speech_recognition as sr
from collections import deque, Counter

# ==========================================
# 1. CONFIGURATION & STOCKAGE
# ==========================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({}, open(LEXICON_PATH, "w", encoding="utf-8"))

# ==========================================
# 2. ETAT SESSION (V2)
# ==========================================
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=30)

# ==========================================
# 3. MOTEUR Φ (amélioré)
# ==========================================
def evolve_phi(phi, excitation):
    phi["phi_m"] = min(1,max(0.1,phi["phi_m"]+excitation*0.15-0.01))
    phi["phi_c"] = min(1,max(0.1,phi["phi_c"]+excitation*0.3-0.03))
    phi["phi_d"] = min(1,max(0.1,phi["phi_d"]+0.02-excitation*0.05))
    return phi

# ==========================================
# 4. MEMOIRE
# ==========================================
def load_lex():
    try:
        return json.load(open(LEXICON_PATH,"r",encoding="utf-8"))
    except:
        return {}

def save_lex(L):
    json.dump(L,open(LEXICON_PATH,"w",encoding="utf-8"),
              indent=2,ensure_ascii=False)

# ==========================================
# 5. APPRENTISSAGE V2
# ==========================================
def learn(text, phi, importance=1.0):
    if not text: return
    words=text.lower().split()
    if len(words)<2: return

    L=load_lex()

    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):
        L.setdefault(a,{})
        L[a][b]=L[a].get(b,0)+energy

    save_lex(L)

# ==========================================
# 6. SEED CONTEXTUEL (NOUVEAU)
# ==========================================
def contextual_seed(L):
    context_words=" ".join(st.session_state.dialog_memory).split()

    candidates=[w for w in context_words if w in L]

    if candidates:
        return Counter(candidates).most_common(1)[0][0]

    return random.choice(list(L.keys()))

# ==========================================
# 7. GENERATION V2
# ==========================================
def oracle_reply(phi):
    L=load_lex()
    if not L:
        return "Mémoire vide. Injectez un signal."

    seed=contextual_seed(L)
    words=[seed]
    used=set(words)

    length=int(8+phi["phi_m"]*35)

    for _ in range(length):
        current=words[-1]
        if current not in L:
            break

        options=L[current]

        # créativité contrôlée
        if random.random()<phi["phi_c"]:
            nxt=random.choices(
                list(options.keys()),
                weights=list(options.values())
            )[0]
        else:
            nxt=max(options,key=options.get)

        # anti boucle
        if nxt in used and random.random()<phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    return " ".join(words).capitalize()+"."

# ==========================================
# 8. INTERFACE
# ==========================================
st.set_page_config(page_title="ORACLE V2",layout="wide")
st.title("🧠 ORACLE V2 — Cognition Contextuelle")

col1,col2=st.columns(2)

# -------- INPUT ----------
with col1:
    txt=st.text_area("Message utilisateur")

    if st.button("⚡ Envoyer"):
        st.session_state.dialog_memory.append(txt)

        excitation=min(1,len(txt)/200)
        st.session_state.phi=evolve_phi(
            st.session_state.phi,
            excitation
        )

        learn(txt,st.session_state.phi,importance=1.2)

        st.success("Signal intégré.")

# -------- OUTPUT ----------
with col2:
    if st.button("💬 Répondre"):
        reply=oracle_reply(st.session_state.phi)

        st.session_state.dialog_memory.append(reply)

        learn(reply,{"phi_m":0.1,"phi_c":0.1,"phi_d":0.1},0.3)

        st.markdown(f"### > {reply}")

# ==========================================
# 9. ETAT Φ
# ==========================================
with st.sidebar:
    st.subheader("Φ Dynamique")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}: {v:.2f}")