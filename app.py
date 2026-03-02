# =====================================================
# 🧠 ORACLE V4.5 Ω — AGENT COGNITIF COMPLET
# Fusion V2 + V3.1 + V4 + V4.5
# =====================================================

import streamlit as st
import random, json, os, math, time, base64
import requests
import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr
from collections import deque, Counter

# =====================================================
# CONFIGURATION
# =====================================================

MEM_FILE="oracle_memory.json"

if not os.path.exists(MEM_FILE):
    json.dump({},open(MEM_FILE,"w"))

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
BRANCH="main"

# =====================================================
# SESSION STATE
# =====================================================

if "phi" not in st.session_state:
    st.session_state.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog" not in st.session_state:
    st.session_state.dialog=deque(maxlen=60)

if "hippocampus" not in st.session_state:
    st.session_state.hippocampus=[]

if "green_state" not in st.session_state:
    st.session_state.green_state=0.0

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep=time.time()

# =====================================================
# GREEN NOISE (HOMEOSTASIS)
# =====================================================

def green_noise(prev):
    return 0.92*prev + 0.08*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state=green_noise(
        st.session_state.green_state
    )
    return abs(st.session_state.green_state)<0.25

# =====================================================
# MEMORY
# =====================================================

def load_memory():
    return json.load(open(MEM_FILE))

def save_memory(M):
    json.dump(M,open(MEM_FILE,"w"),indent=2,ensure_ascii=False)

# =====================================================
# GITHUB AUTO SYNC
# =====================================================

def github_sync():

    with open(MEM_FILE,"rb") as f:
        content=base64.b64encode(f.read()).decode()

    url=f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"
    headers={"Authorization":f"token {GITHUB_TOKEN}"}

    r=requests.get(url,headers=headers)
    sha=None

    if r.status_code==200:
        sha=r.json()["sha"]

    data={
        "message":"🧬 Oracle memory auto-sync",
        "content":content,
        "branch":BRANCH
    }

    if sha:
        data["sha"]=sha

    requests.put(url,headers=headers,json=data)

# =====================================================
# Φ ENGINE (V3.1)
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
# LEARNING — HIPPOCAMPUS
# =====================================================

def learn(text,phi):

    words=text.lower().split()
    if len(words)<2:
        return

    energy=math.sqrt(sum(v*v for v in phi.values()))
    st.session_state.hippocampus.append((words,energy))

    if len(st.session_state.hippocampus)>5 and consolidation_gate():
        consolidate()

def consolidate():

    M=load_memory()

    for words,energy in st.session_state.hippocampus:
        for a,b in zip(words,words[1:]):
            M.setdefault(a,{})
            M[a][b]=M[a].get(b,0)+energy

    save_memory(M)
    st.session_state.hippocampus.clear()

    github_sync()

# =====================================================
# SLEEP CYCLE
# =====================================================

def sleep_cycle():

    M=load_memory()
    new={}

    for w,con in M.items():
        filt={t:v*0.997 for t,v in con.items() if v>1.2}
        if filt:
            new[w]=filt

    save_memory(new)
    st.session_state.last_sleep=time.time()

def auto_sleep():
    if time.time()-st.session_state.last_sleep>180:
        if consolidation_gate():
            sleep_cycle()

auto_sleep()

# =====================================================
# CORTEX (V4)
# =====================================================

def contextual_seed(M):

    ctx=" ".join(st.session_state.dialog).split()
    valid=[w for w in ctx if w in M]

    if valid:
        return Counter(valid).most_common(1)[0][0]

    return random.choice(list(M.keys()))

def logical_layer(seq,M):
    return [w for w in seq if w in M] or seq

def associative_layer(word,M,phi):

    if word not in M:
        return word

    opts=M[word]

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
# GENERATION
# =====================================================

def oracle_reply():

    M=load_memory()
    if not M:
        return "Mémoire vide."

    seed=contextual_seed(M)
    words=[seed]

    length=int(10+st.session_state.phi["phi_m"]*30)

    for _ in range(length):
        filtered=logical_layer(words,M)
        nxt=associative_layer(filtered[-1],M,
                              st.session_state.phi)
        words.append(nxt)

    words=predictive_layer(words,
                           st.session_state.phi)

    return " ".join(words).capitalize()+"."

# =====================================================
# FILE INSERTS (ALL TYPES)
# =====================================================

def read_file(upload):

    if upload.type=="application/pdf":
        reader=PyPDF2.PdfReader(upload)
        return " ".join(p.extract_text() or "" for p in reader.pages)

    if upload.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        d=docx.Document(upload)
        return " ".join(p.text for p in d.paragraphs)

    if upload.type=="text/plain":
        return upload.read().decode()

    if upload.type=="text/csv":
        return pd.read_csv(upload).to_string()

    return ""

# =====================================================
# AUDIO INPUT
# =====================================================

def speech_to_text(audio):

    r=sr.Recognizer()

    with sr.AudioFile(audio) as source:
        data=r.record(source)

    try:
        return r.recognize_google(data)
    except:
        return ""

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V4.5 Ω",page_icon="🧠")

st.title("🧠 ORACLE V4.5 Ω — Agent Cognitif Total")

msg=st.text_input("Parlez à l'Oracle")

file=st.file_uploader(
    "Insérer fichier / audio",
    type=["pdf","docx","txt","csv","wav"]
)

if st.button("Envoyer"):

    if file:
        if file.type=="audio/wav":
            msg=speech_to_text(file)
        else:
            msg=read_file(file)

    if msg:

        st.session_state.dialog.append(msg)

        exc=min(1,len(msg)/200)
        st.session_state.phi=evolve_phi(
            st.session_state.phi,exc
        )

        learn(msg,st.session_state.phi)

        reply=oracle_reply()

        st.session_state.dialog.append(reply)

for m in st.session_state.dialog:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}:{v:.2f}")

    if st.button("🌙 Sommeil forcé"):
        sleep_cycle()
        st.success("Consolidation terminée")

    if st.button("🧬 Sync GitHub"):
        github_sync()
        st.success("Mémoire synchronisée")