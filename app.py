# =====================================================
# 🧠 TTU ORACLE V10 — CORTEX ENFANT
# IA cognitive auto-organisée (sans LLM)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

st.set_page_config(page_title="TTU Oracle V10", layout="wide")

MEM = "memory"

# FIX crash Streamlit Cloud
if os.path.exists(MEM) and not os.path.isdir(MEM):
    os.remove(MEM)

os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "concepts": f"{MEM}/concepts.csv",
    "associations": f"{MEM}/associations.json",
    "cortex": f"{MEM}/cortex.json"
}

# -----------------------------------------------------
# INIT MEMORY
# -----------------------------------------------------

def init():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)

    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept","weight"]).to_csv(FILES["concepts"],index=False)

    if not os.path.exists(FILES["associations"]):
        json.dump({}, open(FILES["associations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({"VS":12.0,"age":0}, open(FILES["cortex"],"w"))

init()

# -----------------------------------------------------
# UNIVERSAL FILE READER (NO DEPENDENCY)
# -----------------------------------------------------

def read_uploaded_file(file):

    name=file.name.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8","ignore")

    if name.endswith(".csv"):
        return pd.read_csv(file).to_string()

    if name.endswith(".pdf"):
        return file.read().decode("latin-1","ignore")

    if name.endswith(".docx"):
        doc=zipfile.ZipFile(io.BytesIO(file.read()))
        xml=doc.read("word/document.xml")
        tree=ET.fromstring(xml)
        return " ".join([n.text for n in tree.iter() if n.text])

    if name.endswith(".xlsx"):
        return pd.read_excel(file).to_string()

    return ""

# -----------------------------------------------------
# LANGUAGE CORE
# -----------------------------------------------------

VOWELS="aeiouyàâéèêëîïôùûüœ"

def clean(text):
    text=text.lower()
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",text)

def tokenize(text):
    return [w for w in clean(text).split() if w]

def syllabify(word):
    syll=[]
    cur=""
    for c in word:
        cur+=c
        if c in VOWELS:
            syll.append(cur)
            cur=""
    if cur:
        syll[-1]+=cur if syll else cur
    return syll

# -----------------------------------------------------
# LOAD / SAVE
# -----------------------------------------------------

def load_frag(): return pd.read_csv(FILES["fragments"])
def save_frag(df): df.to_csv(FILES["fragments"],index=False)

def load_assoc(): return json.load(open(FILES["associations"]))
def save_assoc(a): json.dump(a,open(FILES["associations"],"w"))

def load_cortex(): return json.load(open(FILES["cortex"]))
def save_cortex(c): json.dump(c,open(FILES["cortex"],"w"))

# -----------------------------------------------------
# 🧠 LEARNING (ENFANT)
# -----------------------------------------------------

def learn(text):

    words=tokenize(text)
    pairs=zip(words[:-1],words[1:])

    # ---- fragments
    df=load_frag()
    counter=Counter(words)

    for w,c in counter.items():
        if w in df.fragment.values:
            df.loc[df.fragment==w,"count"]+=c
        else:
            df.loc[len(df)]=[w,c]

    save_frag(df)

    # ---- associations (cognition)
    assoc=load_assoc()

    for a,b in pairs:
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+1

    save_assoc(assoc)

    # ---- cortex growth
    cortex=load_cortex()
    cortex["age"]+=len(words)
    cortex["VS"]=10+np.log1p(cortex["age"])
    save_cortex(cortex)

    return len(words)

# -----------------------------------------------------
# 🧠 THINKING ENGINE
# -----------------------------------------------------

def think(seed,steps=25):

    assoc=load_assoc()

    if seed not in assoc:
        return seed

    sentence=[seed]
    current=seed

    for _ in range(steps):

        nxt=assoc.get(current)
        if not nxt:
            break

        words=list(nxt.keys())
        weights=np.array(list(nxt.values()))
        probs=weights/weights.sum()

        current=np.random.choice(words,p=probs)
        sentence.append(current)

    return " ".join(sentence).capitalize()+"."

# -----------------------------------------------------
# UI
# -----------------------------------------------------

st.title("🧠 TTU ORACLE V10 — CORTEX ENFANT")

cortex=load_cortex()
c1,c2=st.columns(2)
c1.metric("Vitalité Spectrale",round(cortex["VS"],2))
c2.metric("Âge cognitif",cortex["age"])

# -----------------------------------------------------
# LEARN
# -----------------------------------------------------

st.subheader("📥 Nourrir l'enfant")

uploaded_file=st.file_uploader(
    "Importer connaissance",
    type=["txt","csv","pdf","docx","xlsx","json","md"]
)

if uploaded_file:
    text=read_uploaded_file(uploaded_file)
    n=learn(text)
    st.success(f"{n} unités cognitives assimilées")

# -----------------------------------------------------
# CHAT
# -----------------------------------------------------

st.subheader("💬 Dialogue")

prompt=st.text_input("Intention")

if st.button("Penser"):
    if prompt:
        response=think(prompt.split()[0])
        st.write("### Réponse")
        st.write(response)
