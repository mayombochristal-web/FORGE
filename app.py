# ==========================================================
# 🧠 TTU ORACLE V7 — CERVEAU LINGUISTIQUE MULTI-ÉCHELLE
# caractères → syllabes → concepts → contexte
# AUTONOME — STREAMLIT ONLY
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from collections import Counter, defaultdict

DATA_DIR = "ttu_memory"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================================
# UTIL TEXT
# ==========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàâéèêîôùûç]", " ", text)
    return text

def tokenize(text):
    return [w for w in clean_text(text).split() if len(w) > 2]

# ==========================================================
# SYLLABLE ENGINE (heuristique FR)
# ==========================================================

VOWELS = "aeiouyàâéèêîôùû"

def syllabify(word):
    syllables=[]
    current=""

    for c in word:
        current+=c
        if c in VOWELS:
            syllables.append(current)
            current=""

    if current:
        syllables[-1]+=current if syllables else current

    return syllables

# ==========================================================
# EXTRACTION TEXT
# ==========================================================

def extract_text(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except:
        return file.read().decode("latin-1", errors="ignore")

# ==========================================================
# SAVE CSV
# ==========================================================

def save_csv(name, data_dict):

    path = f"{DATA_DIR}/{name}.csv"

    df = pd.DataFrame(
        [(k,v) for k,v in data_dict.items()],
        columns=["element","count"]
    )

    df.to_csv(path,index=False)
    return path

# ==========================================================
# LEVEL 0 — CHARACTERS
# ==========================================================

def analyze_characters(text):
    return Counter(text)

# ==========================================================
# LEVEL 1 — SYLLABLES
# ==========================================================

def analyze_syllables(words):

    syll_counter=Counter()

    for w in words:
        syll_counter.update(syllabify(w))

    return syll_counter

# ==========================================================
# LEVEL 2 — WORDS
# ==========================================================

def analyze_words(words):
    return Counter(words)

# ==========================================================
# LEVEL 3 — FRAGMENTS
# ==========================================================

def analyze_fragments(text):

    sentences=re.split(r"[.!?]",text)
    fragments=[]

    for s in sentences:
        tok=tokenize(s)
        if len(tok)>4:
            fragments.append(" ".join(tok))

    return Counter(fragments)

# ==========================================================
# LEVEL 4 — CONCEPTS (émergents)
# ==========================================================

def analyze_concepts(words):

    clusters=defaultdict(int)

    for w in words:
        key=w[:4]   # racine simple
        clusters[key]+=1

    return clusters

# ==========================================================
# LEVEL 5 — CONTEXT GRAPH
# ==========================================================

def analyze_context(words,window=2):

    relations=Counter()

    for i,w in enumerate(words):
        for j in range(max(0,i-window),min(len(words),i+window+1)):
            if i!=j:
                relations[f"{w}->{words[j]}"]+=1

    return relations

# ==========================================================
# GLOBAL LEARNING PIPELINE
# ==========================================================

def learn(text):

    words=tokenize(text)

    chars=analyze_characters(text)
    syll=analyze_syllables(words)
    word_bank=analyze_words(words)
    frags=analyze_fragments(text)
    concepts=analyze_concepts(words)
    contexts=analyze_context(words)

    paths={}

    paths["characters"]=save_csv("characters",chars)
    paths["syllables"]=save_csv("syllables",syll)
    paths["words"]=save_csv("words",word_bank)
    paths["fragments"]=save_csv("fragments",frags)
    paths["concepts"]=save_csv("concepts",concepts)
    paths["contexts"]=save_csv("contexts",contexts)

    return paths,len(words)

# ==========================================================
# SIMPLE REASONING
# ==========================================================

def oracle_answer(question):

    words=tokenize(question)

    if not words:
        return "Je perçois une intention mais aucun concept stable."

    concept=words[0]

    return (
        f"Le concept '{concept}' apparaît dans un réseau de relations "
        f"où le sens dépend du contexte et des associations apprises. "
        f"La compréhension évolue par accumulation de fragments."
    )

# ==========================================================
# UI
# ==========================================================

st.set_page_config(page_title="🧠 TTU Oracle V7",layout="wide")

st.title("🧠 TTU ORACLE V7 — Cerveau Linguistique")
st.caption("Apprentissage multi-échelle autonome")

uploaded=st.file_uploader(
    "Nourrir l'Oracle (txt, csv, pdf, docx)",
    accept_multiple_files=True
)

if uploaded:

    total_words=0
    last_paths=None

    for f in uploaded:
        text=extract_text(f)
        paths,count=learn(text)
        total_words+=count
        last_paths=paths

    st.success(f"Apprentissage terminé — {total_words} mots intégrés")

    st.subheader("📥 Banques TTU téléchargeables")

    for k,p in last_paths.items():
        with open(p,"rb") as file:
            st.download_button(
                label=f"Télécharger {k}.csv",
                data=file,
                file_name=f"{k}.csv"
            )

# CHAT
question=st.chat_input("Dialogue avec l'Oracle")

if question:
    with st.chat_message("assistant"):
        st.write(oracle_answer(question))
