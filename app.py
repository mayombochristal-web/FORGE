# =====================================================
# 🧠 ORACLE V12 — ORACLE IMMORTEL
# Shadow Memory + TTU Cognitive Engine
# Compatible V11 (upgrade non destructif)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

# ---------- SCIENTIFIC MODULES ----------
try:
    from scipy.signal import stft, coherence
    from scipy.linalg import eigvals
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except:
    SPECTRAL_AVAILABLE = False

# ---------- CONFIG ----------
st.set_page_config(page_title="ORACLE V12 IMMORTEL", layout="wide")

MEM="oracle_memory"
os.makedirs(MEM,exist_ok=True)

FILES={
 "fragments":f"{MEM}/fragments.csv",
 "concepts":f"{MEM}/concepts.csv",
 "relations":f"{MEM}/relations.json",
 "cortex":f"{MEM}/cortex.json"
}

ZIP_PATH="oracle_memory.zip"

# =====================================================
# 🔁 MÉMOIRE IMMORTELLE
# =====================================================

def save_brain_zip():
    with zipfile.ZipFile(ZIP_PATH,"w") as z:
        for f in FILES.values():
            if os.path.exists(f):
                z.write(f)

def load_brain_zip():
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH,"r") as z:
            z.extractall(MEM)

load_brain_zip()

# =====================================================
# INIT
# =====================================================

def init():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)

    if not os.path.exists(FILES["relations"]):
        json.dump({},open(FILES["relations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "age":0,
            "VS":12,
            "timeline":[],
            "temperature":0.4,
            "IS":1.0
        },open(FILES["cortex"],"w"))

init()

# =====================================================
# LOADERS
# =====================================================

def load_json(p):
    return json.load(open(p))

def save_json(p,d):
    json.dump(d,open(p,"w"))

def lazy_csv(path,chunksize=5000):
    chunks=[]
    for c in pd.read_csv(path,chunksize=chunksize):
        chunks.append(c)
    return pd.concat(chunks,ignore_index=True)

# =====================================================
# SESSION SHADOW
# =====================================================

if "shadow" not in st.session_state:
    st.session_state.shadow_frag=lazy_csv(FILES["fragments"])
    st.session_state.shadow_rel=load_json(FILES["relations"])
    st.session_state.shadow_cortex=load_json(FILES["cortex"])
    st.session_state.shadow=True

# =====================================================
# NLP
# =====================================================

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

def semantic_energy(word,freq):
    return 1/(1+freq)

# =====================================================
# LEARN (NGRAM + ENERGY TIMELINE)
# =====================================================

def learn(text,n=3):
    words=tokenize(text)
    if not words:
        return 0

    df=st.session_state.shadow_frag.copy()
    counts=Counter(words)

    for w,c in counts.items():
        mask=df["fragment"]==w
        if mask.any():
            df.loc[mask,"count"]+=c
        else:
            df=pd.concat([df,pd.DataFrame([[w,c]],columns=df.columns)])

    df.reset_index(drop=True,inplace=True)
    df.to_csv(FILES["fragments"],index=False)
    st.session_state.shadow_frag=df

    assoc=st.session_state.shadow_rel

    # NGRAMS
    for i in range(len(words)-n):
        key=" ".join(words[i:i+n-1])
        nxt=words[i+n-1]
        assoc.setdefault(key,{})
        assoc[key][nxt]=assoc[key].get(nxt,0)+1

    save_json(FILES["relations"],assoc)

    # ENERGY TIMELINE
    cortex=st.session_state.shadow_cortex

    for w in words:
        freq=int(df[df.fragment==w]["count"].iloc[0])
        energy=semantic_energy(w,freq)
        cortex["timeline"].append({"w":w,"e":energy})

    cortex["age"]+=len(words)
    cortex["VS"]=10+np.log1p(cortex["age"])

    # sincerity index
    cortex["IS"]=np.mean([t["e"] for t in cortex["timeline"][-200:]])

    save_json(FILES["cortex"],cortex)
    save_brain_zip()

    return len(words)

# =====================================================
# THINK (TEMPÉRATURE TTU)
# =====================================================

def think(seed,steps=30):
    assoc=st.session_state.shadow_rel
    T=st.session_state.shadow_cortex["temperature"]

    if seed not in assoc:
        return "Je dois encore apprendre."

    sent=[seed]
    cur=seed

    for _ in range(steps):
        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)

        p=p**(1/(T+0.01))
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."

# =====================================================
# SPECTRAL SIGNAL
# =====================================================

def build_signal(word):
    tl=st.session_state.shadow_cortex["timeline"]
    return np.array([t["e"] if t["w"]==word else 0 for t in tl])

# =====================================================
# CROSS COHERENCE
# =====================================================

def cross_spectrum(w1,w2):
    if not SPECTRAL_AVAILABLE: return None
    s1=build_signal(w1)
    s2=build_signal(w2)
    if len(s1)<64: return None
    f,Cxy=coherence(s1,s2)
    fig,ax=plt.subplots()
    ax.plot(f,Cxy)
    ax.set_title(f"Cohérence {w1}-{w2}")
    return fig

# =====================================================
# EIGEN THOUGHT ANALYSIS
# =====================================================

def eigen_analysis():
    assoc=st.session_state.shadow_rel
    keys=list(assoc.keys())[:200]
    n=len(keys)
    M=np.zeros((n,n))

    for i,k in enumerate(keys):
        for j,v in enumerate(keys):
            if v in assoc.get(k,{}):
                M[i,j]=assoc[k][v]

    vals=eigvals(M)
    return np.real(vals)

# =====================================================
# UI
# =====================================================

st.title("🧠 ORACLE V12 — IMMORTEL")

ctx=st.session_state.shadow_cortex

c1,c2,c3=st.columns(3)
c1.metric("Vitalité",round(ctx["VS"],2))
c2.metric("Age Cognitif",ctx["age"])
c3.metric("Indice Sincérité",round(ctx["IS"],3))

if ctx["IS"]<0.15:
    st.error("⚠️ BRUIT COGNITIF DETECTÉ")

# temperature
ctx["temperature"]=st.slider(
 "Température cognitive",
 0.05,1.5,ctx["temperature"]
)

# LEARN
st.subheader("📥 Nourrir")
file=st.file_uploader("Texte",type=["txt","csv","docx","pdf"])

if file:
    text=file.read().decode("utf-8","ignore")
    n=learn(text)
    st.success(f"{n} unités apprises")

# CHAT
st.subheader("💬 Dialogue")
prompt=st.text_input("Intention")

if st.button("Penser"):
    st.write(think(prompt.lower()))

# CROSS COHERENCE
if SPECTRAL_AVAILABLE:
    st.subheader("🔬 Résonance conceptuelle")
    words=st.session_state.shadow_frag.fragment.tolist()

    if len(words)>2:
        w1=st.selectbox("Mot 1",words,key=1)
        w2=st.selectbox("Mot 2",words,key=2)

        if st.button("Analyser cohérence"):
            fig=cross_spectrum(w1,w2)
            if fig:
                st.pyplot(fig)

# EIGENVALUES
if st.button("Analyse valeurs propres"):
    vals=eigen_analysis()
    st.write("Modes cognitifs dominants :",vals[:10])
