# =====================================================
# 🧠 ORACLE V4.5 Ω — AGENT COGNITIF BIOLOGIQUE COMPLET
# Base V3.1 + Ghost Cortex + Safe IO + Async Memory
# Architecture TTU-MC³ Stable
# =====================================================

import streamlit as st
import random, json, os, math, time, threading
from collections import deque, Counter
from io import BytesIO

import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr

# =====================================================
# 1. CONFIGURATION
# =====================================================

MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({}, open(LEXICON_PATH, "w", encoding="utf-8"))

# limites sécurité (anti crash Streamlit)
MAX_FILE_MB = 8
MAX_PAGES = 40
MAX_ROWS = 400

# =====================================================
# 2. SESSION STATE
# =====================================================

def init_state():

    defaults = {
        "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
        "dialog_memory":deque(maxlen=60),
        "green_state":0.0,
        "last_sleep":time.time(),
        "ghost_cache":{},
        "hippocampus":[]
    }

    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

init_state()

# =====================================================
# 🧠 3. GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev + 0.08*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state = green_noise(
        st.session_state.green_state
    )
    return abs(st.session_state.green_state)<0.25

# =====================================================
# 4. Φ ENGINE
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
# 5. MÉMOIRE SAFE
# =====================================================

def load_lex():
    try:
        with open(LEXICON_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_lex(L):
    tmp = L.copy()

    # limite taille mémoire (anti AxiosError / IO freeze)
    if len(tmp)>15000:
        tmp=dict(list(tmp.items())[-12000:])

    with open(LEXICON_PATH,"w",encoding="utf-8") as f:
        json.dump(tmp,f,indent=2,ensure_ascii=False)

# =====================================================
# 👻 6. GHOST PRELOAD (V4.5)
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

    st.session_state.ghost_cache=cache

# =====================================================
# 7. HIPPOCAMPUS (mémoire tampon biologique)
# =====================================================

def learn(text,phi,importance=1):

    words=text.lower().split()
    if len(words)<2:
        return

    energy=math.sqrt(sum(v*v for v in phi.values()))*importance
    st.session_state.hippocampus.append((words,energy))

    if len(st.session_state.hippocampus)>5 and consolidation_gate():
        consolidate()

def consolidate():

    L=load_lex()

    for words,energy in st.session_state.hippocampus:
        for a,b in zip(words,words[1:]):
            L.setdefault(a,{})
            L[a][b]=L[a].get(b,0)+energy

    if len(L) > 12000:
    L = dict(list(L.items())[-10000:])

# =====================================================
# 🌙 8. SOMMEIL
# =====================================================

def sleep_cycle():

    L=load_lex()
    new={}

    for w,con in L.items():

        filt={t:v*0.997 for t,v in con.items() if v>1.2}

        if filt:
            new[w]=filt

    save_lex(new)
    st.session_state.last_sleep=time.time()

    return f"🌙 {len(L)-len(new)} synapses oubliées"

def auto_sleep():

    if time.time()-st.session_state.last_sleep>180:
        if consolidation_gate():
            sleep_cycle()

auto_sleep()

# =====================================================
# 🧠 9. CORTEX
# =====================================================

def contextual_seed(L):

    ctx=" ".join(st.session_state.dialog_memory).split()
    valid=[w for w in ctx if w in L]

    if valid:
        return Counter(valid).most_common(1)[0][0]

    return random.choice(list(L.keys()))

def associative_layer(word,L,phi):

    ghost=st.session_state.ghost_cache.get(word)
    if ghost:
        return random.choice(ghost)[0]

    if word not in L:
        return word

    opts=L[word]

    if random.random()<phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

# =====================================================
# 10. GÉNÉRATION
# =====================================================

def oracle_reply():

    L=load_lex()
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(L)

    words=[seed]
    used=set(words)

    length=int(10+st.session_state.phi["phi_m"]*30)

    for _ in range(length):

        nxt=associative_layer(words[-1],L,
                              st.session_state.phi)

        if nxt in used and random.random()<st.session_state.phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    if st.session_state.phi["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."

# =====================================================
# 📂 11. FILE READER SAFE (ANTI AXIOS ERROR)
# =====================================================

def read_file(upload):

    raw=upload.read()

    if len(raw) > MAX_FILE_MB*1024*1024:
        return ""

    try:

        if upload.type=="application/pdf":
            reader=PyPDF2.PdfReader(BytesIO(raw))
            text=[]
            for p in reader.pages[:MAX_PAGES]:
                t=p.extract_text()
                if t:
                    text.append(t)
            return " ".join(text)

        if upload.type.endswith("document"):
            doc=docx.Document(BytesIO(raw))
            return " ".join(p.text for p in doc.paragraphs[:400])

        if upload.type=="text/plain":
            return raw.decode("utf-8","ignore")

        if upload.type=="text/csv":
            df=pd.read_csv(BytesIO(raw))
            return df.head(MAX_ROWS).to_string()

    except:
        pass

    return ""

# =====================================================
# 🎤 AUDIO
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
# 12. UI
# =====================================================

st.set_page_config(page_title="ORACLE V4.5 Ω",page_icon="🧠")
st.title("🧠 ORACLE V4.5 Ω — Cognitive Agent")

msg_input=st.text_input("Parlez à l'Oracle")

file=st.file_uploader(
    "Insérer fichier / audio",
    type=["pdf","docx","txt","csv","wav"]
)

# =====================================================
# PIPELINE
# =====================================================

if st.button("Envoyer"):

    msg=""

    if file:
        if file.type=="audio/wav":
            msg=speech_to_text(file)
        else:
            msg=read_file(file)
    else:
        msg=msg_input

    if msg:

        ghost_preload(msg)

        st.session_state.dialog_memory.append(msg)

        exc=min(1,len(msg)/200)

        st.session_state.phi=evolve_phi(
            st.session_state.phi,exc
        )

        learn(msg,st.session_state.phi)

        reply=oracle_reply()

        st.session_state.dialog_memory.append(reply)

        learn(reply,
              {"phi_m":0.1,"phi_c":0.1,"phi_d":0.1},
              0.3)

# =====================================================
# DISPLAY
# =====================================================

for m in st.session_state.dialog_memory:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    size=os.path.getsize(LEXICON_PATH)/1024
    st.success(f"Mémoire : {size:.2f} KB")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}: {v:.2f}")

    if st.button("🌙 Sommeil forcé"):
        st.warning(sleep_cycle())
        st.rerun()