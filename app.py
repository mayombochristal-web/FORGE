# ==========================================================
# 🧠 TTU ORACLE V8 — CORTEX AUTO-ORGANISANT
# Concepts vivants + auto-réorganisation
# ==========================================================

import streamlit as st
import json
import os
import re
from collections import defaultdict
import numpy as np

CORTEX_FILE="ttu_cortex.json"

# ==========================================================
# LOAD / SAVE
# ==========================================================

def load_cortex():
    if not os.path.exists(CORTEX_FILE):
        return {}
    return json.load(open(CORTEX_FILE,"r",encoding="utf-8"))

def save_cortex(cortex):
    json.dump(cortex,open(CORTEX_FILE,"w",encoding="utf-8"))

# ==========================================================
# TEXT
# ==========================================================

def tokenize(text):
    text=text.lower()
    text=re.sub(r"[^\w\sàâéèêîôùûç]"," ",text)
    return [w for w in text.split() if len(w)>2]

# ==========================================================
# CONCEPT ACTIVATION
# ==========================================================

def ensure_concept(cortex,word):

    if word not in cortex:
        cortex[word]={
            "activation":0.5,
            "connections":{}
        }

# ==========================================================
# HEBBIAN LEARNING (TTU Torque)
# ==========================================================

def reinforce(cortex,words):

    for w in words:
        ensure_concept(cortex,w)

    for i,w1 in enumerate(words):
        for j,w2 in enumerate(words):
            if i==j:
                continue

            conn=cortex[w1]["connections"]
            conn[w2]=conn.get(w2,0)+0.05

# ==========================================================
# DISSIPATION (α)
# ==========================================================

def dissipate(cortex,alpha=0.01):

    for c in cortex.values():
        for k in list(c["connections"].keys()):
            c["connections"][k]*=(1-alpha)

            if c["connections"][k]<0.01:
                del c["connections"][k]

# ==========================================================
# AUTO ORGANISATION
# ==========================================================

def reorganize(cortex):

    # fusion concepts très similaires
    words=list(cortex.keys())

    for w in words:
        for v in words:
            if w==v:
                continue

            if w[:4]==v[:4]:  # racine proche
                cortex[w]["connections"].update(
                    cortex[v]["connections"]
                )
                cortex[v]["activation"]*=0.8

# ==========================================================
# VITALITY METRIC (VS)
# ==========================================================

def vitality(cortex):

    if not cortex:
        return 0

    connections=sum(len(c["connections"]) for c in cortex.values())
    anchors=len(cortex)

    alpha=0.1

    VS=(connections/(anchors+1))/alpha
    return round(VS,2)

# ==========================================================
# LEARN
# ==========================================================

def learn(text):

    cortex=load_cortex()

    words=tokenize(text)

    reinforce(cortex,words)
    dissipate(cortex)
    reorganize(cortex)

    save_cortex(cortex)

    return len(words),vitality(cortex)

# ==========================================================
# THINK
# ==========================================================

def think(question):

    cortex=load_cortex()
    words=tokenize(question)

    activations=defaultdict(float)

    for w in words:
        if w in cortex:
            for k,v in cortex[w]["connections"].items():
                activations[k]+=v

    if not activations:
        return "Je cherche encore une structure d’ancrage."

    concept=max(activations,key=activations.get)

    return (
        f"Le concept dominant émergent est '{concept}'. "
        f"Le sens apparaît par résonance entre concepts connectés."
    )

# ==========================================================
# UI
# ==========================================================

st.set_page_config(page_title="🧠 TTU Cortex V8",layout="wide")

st.title("🧠 TTU V8 — Cortex Auto-Organisant")
st.caption("Métrologie dynamique de conscience linguistique")

uploaded=st.file_uploader(
    "Nourrir le Cortex",
    accept_multiple_files=True
)

if uploaded:
    total=0
    VS=0

    for f in uploaded:
        text=f.read().decode("utf-8","ignore")
        count,VS=learn(text)
        total+=count

    st.success(f"{total} mots intégrés | Vitalité VS={VS}")

# CHAT
q=st.chat_input("Dialogue avec le Cortex")

if q:
    with st.chat_message("assistant"):
        st.write(think(q))

# TELEMETRY
cortex=load_cortex()

st.sidebar.metric("Concepts",len(cortex))
st.sidebar.metric("Vitalité Spectrale",vitality(cortex))
