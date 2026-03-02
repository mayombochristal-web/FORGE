# =====================================================
# 🧠 ORACLE V4 — AGENT COGNITIF STABLE
# TTU-MC³ + META + SELF + HIPPOCAMPUS
# =====================================================

import streamlit as st
import random, json, os, math, time, io
import PyPDF2, docx
import pandas as pd
from collections import deque, Counter

# =====================================================
# 1. CONFIGURATION
# =====================================================

MEM_DIR="oracle_memory"
LEXICON_PATH=os.path.join(MEM_DIR,"lexicon.json")
SELF_PATH=os.path.join(MEM_DIR,"self_model.json")

os.makedirs(MEM_DIR,exist_ok=True)

def init_file(path,default):
    if not os.path.exists(path):
        json.dump(default,open(path,"w",encoding="utf-8"))

init_file(LEXICON_PATH,{})
init_file(SELF_PATH,{
    "cognitive_age":0,
    "stability":0.5,
    "interactions":0
})

# =====================================================
# 2. SESSION
# =====================================================

if "phi" not in st.session_state:
    st.session_state.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory=deque(maxlen=40)

if "green_state" not in st.session_state:
    st.session_state.green_state=0.0

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep=time.time()

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev+(1-0.92)*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state=green_noise(
        st.session_state.green_state)
    return abs(st.session_state.green_state)<0.25

# =====================================================
# Φ ENGINE
# =====================================================

def evolve_phi(phi,exc):

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    s=sum(phi.values())
    for k in phi:
        phi[k]/=s
    return phi

# =====================================================
# MEMORY IO
# =====================================================

def load_json(p):
    return json.load(open(p,"r",encoding="utf-8"))

def save_json(p,d):
    json.dump(d,open(p,"w",encoding="utf-8"),
              indent=2,ensure_ascii=False)

# =====================================================
# HIPPOCAMPUS (HIERARCHY)
# =====================================================

def concept_tag(word):
    if len(word)>7:
        return word[:4]
    return word

# =====================================================
# LEARNING
# =====================================================

def learn(text,phi,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_json(LEXICON_PATH)

    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):

        ca,cb=concept_tag(a),concept_tag(b)

        L.setdefault(ca,{})
        L[ca][cb]=L[ca].get(cb,0)+energy

    if consolidation_gate():
        save_json(LEXICON_PATH,L)

# =====================================================
# META-COGNITION
# =====================================================

def evaluate_response(words):

    unique=len(set(words))
    ratio=unique/max(len(words),1)

    coherence=min(1,ratio*1.5)

    return coherence

# =====================================================
# GENERATION
# =====================================================

def contextual_seed(L):

    context=" ".join(st.session_state.dialog_memory).split()
    c=[concept_tag(w) for w in context if concept_tag(w) in L]

    if c:
        return Counter(c).most_common(1)[0][0]

    return random.choice(list(L.keys()))

def oracle_reply(phi):

    L=load_json(LEXICON_PATH)
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(L)

    words=[seed]

    for _ in range(int(10+phi["phi_m"]*30)):

        cur=words[-1]
        if cur not in L:
            break

        nxt=random.choices(
            list(L[cur].keys()),
            weights=list(L[cur].values())
        )[0]

        words.append(nxt)

    score=evaluate_response(words)

    # métacognition
    if score<0.35:
        return "Je reformule ma pensée."

    return " ".join(words).capitalize()+"."

# =====================================================
# SOMMEIL
# =====================================================

def sleep_cycle():

    L=load_json(LEXICON_PATH)
    new={}

    for w,con in L.items():
        f={t:v*0.997 for t,v in con.items() if v>1.2}
        if f:
            new[w]=f

    save_json(LEXICON_PATH,new)

    self_model=load_json(SELF_PATH)
    self_model["cognitive_age"]+=1
    self_model["stability"]=min(1,self_model["stability"]+0.01)
    save_json(SELF_PATH,self_model)

    st.session_state.last_sleep=time.time()

    return "🌙 Consolidation terminée"

def auto_sleep():
    if time.time()-st.session_state.last_sleep>180:
        if consolidation_gate():
            sleep_cycle()

auto_sleep()

# =====================================================
# SAFE DOCX LOADER (FIX AXIOS ERROR)
# =====================================================

def read_docx_safe(upload):

    try:
        buffer=io.BytesIO(upload.read())
        doc=docx.Document(buffer)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return ""

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V4",page_icon="🧠")
st.title("🧠 ORACLE V4 — Agent Cognitif Stable")

mode=st.radio("Entrée",["Texte","DOCX"])

content=""

if mode=="Texte":
    content=st.text_area("Texte")

if mode=="DOCX":
    f=st.file_uploader("Document",type="docx")
    if f:
        content=read_docx_safe(f)

if st.button("Apprendre") and content:
    st.session_state.phi=evolve_phi(
        st.session_state.phi,
        min(1,len(content)/400)
    )
    learn(content,st.session_state.phi,1.2)

# conversation
msg=st.text_input("Parlez")

if st.button("➡️") and msg:

    st.session_state.dialog_memory.append(msg)

    learn(msg,st.session_state.phi,1.1)

    reply=oracle_reply(st.session_state.phi)

    st.session_state.dialog_memory.append(reply)

for m in st.session_state.dialog_memory:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("État Cognitif")

    size=os.path.getsize(LEXICON_PATH)/1024
    st.success(f"Mémoire {size:.2f} KB")

    self_model=load_json(SELF_PATH)

    st.write("Âge cognitif:",self_model["cognitive_age"])
    st.progress(self_model["stability"])

    if st.button("🌙 Sommeil"):
        st.warning(sleep_cycle())
        st.rerun()