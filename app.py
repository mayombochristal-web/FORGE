# =====================================================
# 🧠 ORACLE V4.5 Ω — AGENT COGNITIF COMPLET FINAL
# Fusion V2 + V3.1 + V4 + V4.5 + V5 + V5.5 Ghost Cortex
# =====================================================

import streamlit as st
import random, json, os, math, time, base64, threading
import requests
import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr
from collections import deque, Counter
from io import BytesIO

# =====================================================
# CONFIGURATION
# =====================================================

MEM_FILE="oracle_memory.json"

if not os.path.exists(MEM_FILE):
    json.dump({},open(MEM_FILE,"w",encoding="utf-8"))

GITHUB_TOKEN=st.secrets["GITHUB_TOKEN"]
GITHUB_REPO=st.secrets["GITHUB_REPO"]
BRANCH="main"

# =====================================================
# SESSION STATE
# =====================================================

defaults={
    "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
    "dialog":deque(maxlen=60),
    "hippocampus":[],
    "green_state":0.0,
    "last_sleep":time.time(),
    "ghost_cache":{}
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev+0.08*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state=green_noise(
        st.session_state.green_state
    )
    return abs(st.session_state.green_state)<0.25

# =====================================================
# MEMORY
# =====================================================

def load_memory():
    try:
        with open(MEM_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_memory(M):
    with open(MEM_FILE,"w",encoding="utf-8") as f:
        json.dump(M,f,indent=2,ensure_ascii=False)

# =====================================================
# GITHUB SYNC (SAFE ASYNC)
# =====================================================

def github_sync():
    try:
        with open(MEM_FILE,"rb") as f:
            content=base64.b64encode(f.read()).decode()

        url=f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"
        headers={"Authorization":f"token {GITHUB_TOKEN}"}

        r=requests.get(url,headers=headers,timeout=10)
        sha=r.json()["sha"] if r.status_code==200 else None

        data={
            "message":"🧬 Oracle memory auto-sync",
            "content":content,
            "branch":BRANCH
        }

        if sha:
            data["sha"]=sha

        requests.put(url,headers=headers,json=data,timeout=10)

    except Exception as e:
        st.warning(f"Sync GitHub échoué : {e}")

def async_sync():
    threading.Thread(target=github_sync,daemon=True).start()

# =====================================================
# 👻 GHOST PRELOAD
# =====================================================

def ghost_preload(text):
    try:
        M=load_memory()
        preload={}
        for w in text.lower().split():
            if w in M:
                preload[w]=sorted(
                    M[w].items(),
                    key=lambda x:-x[1]
                )[:5]
        st.session_state.ghost_cache=preload
    except:
        st.session_state.ghost_cache={}

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

    M=load_memory()

    for words,energy in st.session_state.hippocampus:
        for a,b in zip(words,words[1:]):
            M.setdefault(a,{})
            M[a][b]=M[a].get(b,0)+energy

    save_memory(M)
    st.session_state.hippocampus.clear()
    async_sync()

# =====================================================
# SLEEP
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

# =====================================================
# CORTEX
# =====================================================

def contextual_seed(M):

    if not M:
        return "oracle"

    ctx=" ".join(st.session_state.dialog).split()
    valid=[w for w in ctx if w in M]

    if valid:
        return Counter(valid).most_common(1)[0][0]

    return random.choice(list(M.keys()))

def associative_layer(word,M,phi):

    ghost=st.session_state.ghost_cache.get(word)
    if ghost:
        return random.choice(ghost)[0]

    if word not in M:
        return word

    opts=M[word]

    if random.random()<phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

def oracle_reply():

    M=load_memory()
    if not M:
        return "Mémoire vide."

    seed=contextual_seed(M)
    words=[seed]

    length=int(10+st.session_state.phi["phi_m"]*30)

    for _ in range(length):
        nxt=associative_layer(
            words[-1],M,st.session_state.phi
        )
        words.append(nxt)

    return " ".join(words).capitalize()+"."

# =====================================================
# 👻 FILE READING SAFE (ANTI AXIOS)
# =====================================================

def ghost_read_file(upload):

    try:
        raw=upload.read()
        if not raw:
            return ""

        if upload.type=="application/pdf":
            reader=PyPDF2.PdfReader(BytesIO(raw))
            return " ".join(
                (p.extract_text() or "")
                for p in reader.pages[:50]
            )

        if upload.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc=docx.Document(BytesIO(raw))
            return " ".join(p.text for p in doc.paragraphs[:500])

        if upload.type=="text/plain":
            return raw.decode("utf-8","ignore")

        if upload.type=="text/csv":
            df=pd.read_csv(BytesIO(raw))
            return df.head(500).to_string()

    except Exception as e:
        st.warning(f"Lecture sécurisée interrompue : {e}")

    return ""

# =====================================================
# AUDIO
# =====================================================

def speech_to_text(audio):
    try:
        r=sr.Recognizer()
        with sr.AudioFile(audio) as source:
            data=r.record(source)
        return r.recognize_google(data)
    except:
        return ""

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V4.5 Ω",page_icon="🧠")
st.title("🧠 ORACLE V4.5 Ω — Agent Cognitif Total")

msg_input=st.text_input("Parlez à l'Oracle")

file=st.file_uploader(
    "Insérer fichier / audio",
    type=["pdf","docx","txt","csv","wav"]
)

# =====================================================
# PIPELINE
# =====================================================

if st.button("Envoyer"):

    progress=st.progress(0,text="🧠 Activation corticale...")
    status=st.empty()

    msg=""

    if file:
        status.info("📂 Lecture du fichier...")
        progress.progress(20)

        if file.type=="audio/wav":
            msg=speech_to_text(file)
        else:
            msg=ghost_read_file(file)
    else:
        msg=msg_input

    progress.progress(40)

    status.info("🔎 Analyse cognitive...")
    time.sleep(0.2)

    progress.progress(60)

    if msg:
        ghost_preload(msg)

        st.session_state.dialog.append(msg)

        excitation=min(1,len(msg)/200)
        st.session_state.phi=evolve_phi(
            st.session_state.phi,excitation
        )

        learn(msg,st.session_state.phi)

    progress.progress(75)

    status.info("💭 Génération pensée...")
    reply=oracle_reply()
    st.session_state.dialog.append(reply)

    progress.progress(90)

    status.info("☁️ Synchronisation mémoire...")
    async_sync()

    progress.progress(100)
    status.success("✅ Oracle mis à jour")
    time.sleep(1)
    progress.empty()

# =====================================================
# DISPLAY
# =====================================================

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
        async_sync()
        st.success("Mémoire synchronisée")