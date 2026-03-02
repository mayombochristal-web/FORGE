# =====================================================
# 🧠 ORACLE CORE V6 Ω — Biological Persistent Cortex
# =====================================================

import os, json, random, math, time, base64, requests
from collections import Counter
from io import BytesIO
import streamlit as st

import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"
MEMORY_FILE = f"{DATA_DIR}/memory.json"

MAX_FILE_MB = 8
MAX_PAGES = 40
MAX_ROWS = 400

os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================
# SESSION INIT
# =====================================================

def init_state():

    defaults = {
        "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
        "green_state":0.0,
        "last_sleep":time.time(),
        "ghost_cache":{},
        "hippocampus":[],
        "memory_dirty":False,
        "last_sync_check":0
    }

    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

init_state()

# =====================================================
# MEMORY
# =====================================================

def load_memory():

    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE,"w",encoding="utf-8") as f:
            json.dump({"messages":[],"lexicon":{}},f)

    with open(MEMORY_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

def save_memory(mem):

    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        json.dump(mem,f,indent=2,ensure_ascii=False)

    st.session_state.memory_dirty=True


# =====================================================
# MEMORY LIMITER (ANTI JSON WALL)
# =====================================================

def trim_memory():

    mem = load_memory()

    if len(mem["messages"]) > 400:
        mem["messages"] = mem["messages"][-300:]

    L = mem["lexicon"]
    if len(L) > 15000:
        mem["lexicon"] = dict(list(L.items())[-12000:])

    save_memory(mem)

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev + 0.08*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state = green_noise(
        st.session_state.green_state
    )
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
# 👻 GHOST CORTEX
# =====================================================

def ghost_preload(text):

    mem=load_memory()
    L=mem["lexicon"]
    cache={}

    for w in text.lower().split():
        if w in L:
            cache[w]=sorted(
                L[w].items(),
                key=lambda x:-x[1]
            )[:5]

    st.session_state.ghost_cache=cache

def ghost_warm_start():

    mem=load_memory()
    L=mem["lexicon"]

    if not L:
        return

    sample=list(L.keys())[:25]
    cache={}

    for w in sample:
        cache[w]=list(L[w].items())[:3]

    st.session_state.ghost_cache=cache

ghost_warm_start()

# =====================================================
# HIPPOCAMPUS
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

    mem=load_memory()
    L=mem["lexicon"]

    for words,energy in st.session_state.hippocampus:
        for a,b in zip(words,words[1:]):
            L.setdefault(a,{})
            L[a][b]=L[a].get(b,0)+energy

    st.session_state.hippocampus.clear()

    save_memory(mem)
    trim_memory()

# =====================================================
# SOMMEIL
# =====================================================

def sleep_cycle():

    mem=load_memory()
    L=mem["lexicon"]
    new={}

    for w,con in L.items():
        filt={t:v*0.997 for t,v in con.items() if v>1.2}
        if filt:
            new[w]=filt

    mem["lexicon"]=new
    save_memory(mem)
    st.session_state.last_sleep=time.time()

def auto_sleep():

    if time.time()-st.session_state.last_sleep>180:
        if consolidation_gate():
            sleep_cycle()

# =====================================================
# CORTEX
# =====================================================

def associative_layer(word,L):

    ghost=st.session_state.ghost_cache.get(word)
    if ghost:
        return random.choice(ghost)[0]

    if word not in L:
        return word

    opts=L[word]

    if random.random()<st.session_state.phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

def generate_reply():

    mem=load_memory()
    L=mem["lexicon"]

    if not L:
        return "Mémoire vide."

    seed=random.choice(list(L.keys()))
    words=[seed]

    for _ in range(int(10+30*st.session_state.phi["phi_m"])):
        nxt=associative_layer(words[-1],L)
        words.append(nxt)

    if st.session_state.phi["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."

# =====================================================
# FILE INSERTS
# =====================================================

def read_file(upload):

    raw=upload.read()
    if len(raw)>MAX_FILE_MB*1024*1024:
        return ""

    try:
        if upload.type=="application/pdf":
            reader=PyPDF2.PdfReader(BytesIO(raw))
            return " ".join(
                p.extract_text() or ""
                for p in reader.pages[:MAX_PAGES]
            )

        if upload.type.endswith("document"):
            doc=docx.Document(BytesIO(raw))
            return " ".join(p.text for p in doc.paragraphs)

        if upload.type=="text/plain":
            return raw.decode("utf-8","ignore")

        if upload.type=="text/csv":
            df=pd.read_csv(BytesIO(raw))
            return df.head(MAX_ROWS).to_string()

    except:
        pass

    return ""

# =====================================================
# AUDIO
# =====================================================

def speech_to_text(file):
    try:
        r=sr.Recognizer()
        with sr.AudioFile(file) as src:
            audio=r.record(src)
        return r.recognize_google(audio)
    except:
        return ""

# =====================================================
# PIPELINE
# =====================================================

def process_input(text):

    auto_sleep()

    ghost_preload(text)

    exc=min(1,len(text)/200)
    st.session_state.phi=evolve_phi(
        st.session_state.phi,exc
    )

    learn(text,st.session_state.phi)

    reply=generate_reply()

    learn(reply,{"phi_m":0.1,"phi_c":0.1,"phi_d":0.1})

    mem=load_memory()
    mem["messages"].append({"role":"user","content":text})
    mem["messages"].append({"role":"assistant","content":reply})

    save_memory(mem)