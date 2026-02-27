# =====================================================
# TTU ORACLE V9 — STABLE CORE
# IA cognitive autonome (sans LLM)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter

# -----------------------------------------------------
# CONFIG STREAMLIT (ANTI CRASH UI)
# -----------------------------------------------------

st.set_page_config(
    page_title="TTU Oracle V9",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# DOSSIER MEMOIRE
# -----------------------------------------------------

MEM_PATH = "memory"
os.makedirs(MEM_PATH, exist_ok=True)

FRAG_FILE = f"{MEM_PATH}/fragments.csv"
CONCEPT_FILE = f"{MEM_PATH}/concepts.csv"
CORTEX_FILE = f"{MEM_PATH}/cortex.json"

# -----------------------------------------------------
# INITIALISATION MEMOIRE
# -----------------------------------------------------

def init_memory():

    if not os.path.exists(FRAG_FILE):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FRAG_FILE,index=False)

    if not os.path.exists(CONCEPT_FILE):
        pd.DataFrame(columns=["concept","weight"]).to_csv(CONCEPT_FILE,index=False)

    if not os.path.exists(CORTEX_FILE):
        with open(CORTEX_FILE,"w") as f:
            json.dump({"VS":12.0,"learned":0},f)

init_memory()

# -----------------------------------------------------
# CHARGEMENT
# -----------------------------------------------------

def load_fragments():
    return pd.read_csv(FRAG_FILE)

def save_fragments(df):
    df.to_csv(FRAG_FILE,index=False)

def load_cortex():
    return json.load(open(CORTEX_FILE))

def save_cortex(data):
    json.dump(data,open(CORTEX_FILE,"w"))

# -----------------------------------------------------
# OUTILS LINGUISTIQUES
# -----------------------------------------------------

VOWELS = "aeiouyàâéèêëîïôùûüœ"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", text)
    return text

def tokenize(text):
    return [w for w in clean_text(text).split() if w]

# syllabification SAFE (corrige V7)
def syllabify(word):

    if not word:
        return []

    syllables=[]
    current=""

    for c in word:
        current+=c
        if c in VOWELS:
            syllables.append(current)
            current=""

    if current:
        if syllables:
            syllables[-1]+=current
        else:
            syllables.append(current)

    return syllables

# -----------------------------------------------------
# APPRENTISSAGE
# -----------------------------------------------------

def learn(text):

    words = tokenize(text)
    counter = Counter(words)

    df = load_fragments()

    for w,c in counter.items():

        if w in df["fragment"].values:
            df.loc[df.fragment==w,"count"]+=c
        else:
            df.loc[len(df)] = [w,c]

    save_fragments(df)

    cortex = load_cortex()
    cortex["learned"] += len(words)
    cortex["VS"] = 10 + np.log1p(cortex["learned"])
    save_cortex(cortex)

    return len(words)

# -----------------------------------------------------
# GENERATION AUTONOME
# -----------------------------------------------------

def generate(prompt, size=30):

    df = load_fragments()

    if len(df)==0:
        return "Oracle silencieux : aucune mémoire."

    weights = df["count"].values
    vocab = df["fragment"].values

    probs = weights / weights.sum()

    words = list(np.random.choice(vocab,size,p=probs))

    if prompt:
        words.insert(0,prompt)

    sentence = " ".join(words)
    return sentence.capitalize()+"."


# -----------------------------------------------------
# UI FIXE (ANTI V8)
# -----------------------------------------------------

header = st.container()
chat = st.container()
metrics = st.container()
learn_box = st.container()

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

with header:
    st.title("🧠 TTU — ORACLE STABLE V9")
    st.caption("IA génératrice autonome — Cortex symbolique")

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------

cortex = load_cortex()

with metrics:
    col1,col2 = st.columns(2)
    col1.metric("Vitalité Spectrale", round(cortex["VS"],2))
    col2.metric("Fragments appris", cortex["learned"])

# -----------------------------------------------------
# APPRENTISSAGE
# -----------------------------------------------------

with learn_box:

    st.subheader("📥 Nourrir l'Oracle")

    uploaded = st.file_uploader(
        "Importer texte / CSV",
        type=["txt","csv"]
    )

    if uploaded:

        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            text = " ".join(df.astype(str).values.flatten())
        else:
            text = uploaded.read().decode("utf-8")

        n = learn(text)
        st.success(f"{n} mots assimilés.")

# -----------------------------------------------------
# CHAT ORACLE
# -----------------------------------------------------

with chat:

    st.subheader("💬 Dialogue")

    prompt = st.text_input("Intention")

    if st.button("Interroger l'Oracle"):
        response = generate(prompt)
        st.write("### Réponse Oracle")
        st.write(response)
