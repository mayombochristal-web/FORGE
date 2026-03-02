# =====================================================
# 🧠 ORACLE V3.2 — BIOLOGICAL COGNITIVE SYSTEM
# Green Noise + Identity + Ghost Prediction
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

MEM_DIR="oracle_memory"
LEXICON_PATH=os.path.join(MEM_DIR,"lexicon.json")

os.makedirs(MEM_DIR,exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({},open(LEXICON_PATH,"w",encoding="utf-8"))

# =====================================================
# 2. SESSION STATE
# =====================================================

if "phi" not in st.session_state:
    st.session_state.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory=deque(maxlen=40)

if "green_state" not in st.session_state:
    st.session_state.green_state=0.0

if "ghost_cache" not in st.session_state:
    st.session_state.ghost_cache=[]

if "identity_entropy" not in st.session_state:
    st.session_state.identity_entropy=0.5

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep=time.time()

# =====================================================
# 🧠 3. GREEN NOISE
# =====================================================

def green_noise(prev):
    alpha=0.92
    return alpha*prev+(1-alpha)*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state=green_noise(
        st.session_state.green_state
    )
    return abs(st.session_state.green_state)<0.25

# =====================================================
# 4. Φ ENGINE (BIOLOGICAL UPDATE)
# =====================================================

def evolve_phi(phi,excitation):

    # influence Green Noise
    noise=st.session_state.green_state
    phi["phi_c"]+=noise*0.05

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+excitation*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+excitation*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-excitation*0.05))

    total=sum(phi.values())
    for k in phi:
        phi[k]/=total

    # identité émergente
    st.session_state.identity_entropy+=random.uniform(-0.01,0.01)
    st.session_state.identity_entropy=max(
        0,min(1,st.session_state.identity_entropy)
    )

    return phi

# =====================================================
# 5. MEMORY
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
# 6. HIPPOCAMPUS LEARNING
# =====================================================

def learn(text,phi,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_lex()

    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):
        L.setdefault(a,{})
        L[a][b]=L[a].get(b,0)+energy

    if consolidation_gate():
        save_lex(L)

# =====================================================
# 🧠 7. FILE READER → HIPPOCAMPUS
# =====================================================

def read_file(upload):

    text=""

    if upload.name.endswith(".pdf"):
        reader=PyPDF2.PdfReader(upload)
        for p in reader.pages:
            text+=p.extract_text() or ""

    elif upload.name.endswith(".docx"):
        doc=docx.Document(upload)
        text="\n".join(p.text for p in doc.paragraphs)

    return text

# =====================================================
# 8. PROCESS INPUT (GHOST ANTICIPATION)
# =====================================================

def process_input(text):

    ghost_factor=len(st.session_state.ghost_cache)/5
    exc=min(1,(len(text)/200)+ghost_factor*0.2)

    st.session_state.phi=evolve_phi(
        st.session_state.phi,exc
    )

    learn(text,st.session_state.phi,1.1)

    # mémoire fantôme (anticipation)
    st.session_state.ghost_cache.append(text)
    if len(st.session_state.ghost_cache)>10:
        st.session_state.ghost_cache.pop(0)

# =====================================================
# 9. SLEEP
# =====================================================

def sleep_cycle():

    L=load_lex()
    new_L={}

    for w,con in L.items():
        filtered={t:v*0.997 for t,v in con.items() if v>1.2}
        if filtered:
            new_L[w]=filtered

    save_lex(new_L)

    st.session_state.last_sleep=time.time()

    return f"🌙 Sommeil terminé — {len(L)-len(new_L)} synapses oubliées"

# =====================================================
# 10. AUTO FATIGUE
# =====================================================

def auto_sleep():

    mem=load_lex()

    if len(mem)>800:
        sleep_cycle()

auto_sleep()

# =====================================================
# 11. GENERATION
# =====================================================

def oracle_reply(phi):

    L=load_lex()
    if not L:
        return "Mémoire vide."

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
# 12. UI
# =====================================================

st.set_page_config(page_title="ORACLE V3.2",page_icon="🧠")

st.title("🧠 ORACLE V3.2 — Cognitive Biological Engine")

user_msg=st.text_input("Parlez à l'Oracle")

uploaded=st.file_uploader("Insérer document")

audio_text=None  # placeholder perception audio

if uploaded:
    file_text=read_file(uploaded)
    learn(file_text,st.session_state.phi)   # hippocampus

# perception complète
if audio_text:
    process_input(audio_text)

if st.button("➡️") and user_msg:

    st.session_state.dialog_memory.append(user_msg)

    process_input(user_msg)

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

    st.write(f"Identity entropy : {st.session_state.identity_entropy:.2f}")

    if st.button("🌙 Sommeil forcé"):
        st.warning(sleep_cycle())
        st.rerun()