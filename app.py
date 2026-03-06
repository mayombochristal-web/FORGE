# =====================================================
# 🧠 ORACLE V13 LAB
# Fusion ORACLE V11 + V13 Engine
# Shadow State + Ghost Memory + Spectral Semantics
# =====================================================

import streamlit as st
import numpy as np
import json
import datetime
import os
import re
import pandas as pd

from oracle_v13_engine import respond, load_brain

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE V13 LAB", layout="wide")

BRAIN_FILE="oracle_v13_brain.json"


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "brain_shadow" not in st.session_state:

    st.session_state.brain_shadow=load_brain()


# =====================================================
# TOKENIZER
# =====================================================

def tokenize(t):

    t=re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

    return [w for w in t.split() if len(w)>1]


# =====================================================
# DIAGNOSTIC IA
# =====================================================

def diagnose():

    brain=st.session_state.brain_shadow

    age=brain["cortex"]["cognitive_age"]

    vocab=len(brain["lexical_memory"]["tokens"])

    if vocab<50:
        return "🧠 J'ai besoin d'apprendre davantage."

    if vocab<200:
        return "🧠 Mon vocabulaire commence à émerger."

    if vocab<1000:
        return "🧠 Mon intelligence linguistique se développe."

    return "🧠 Intelligence linguistique avancée."


# =====================================================
# METRICS
# =====================================================

def association_density():

    brain=st.session_state.brain_shadow

    rel=brain["language_model"]["bigram"]

    links=sum(len(v) for v in rel.values())

    vocab=len(rel)

    if vocab==0:
        return 0

    return round(links/vocab,2)


def semantic_entropy():

    brain=st.session_state.brain_shadow

    ent=[v["entropy"] for v in brain["lexical_memory"]["tokens"].values()]

    if not ent:
        return 0

    return round(np.mean(ent),3)


# =====================================================
# FILE LEARNING
# =====================================================

def read_file(file):

    name=file.name.lower()

    try:

        if name.endswith(".txt"):
            return file.read().decode()

        if name.endswith(".csv"):
            return pd.read_csv(file).to_string()

        if name.endswith(".xlsx"):
            return pd.read_excel(file).to_string()

    except:

        return ""

    return ""


# =====================================================
# SPECTRAL ANALYSIS
# =====================================================

def spectral_analysis(word):

    brain=st.session_state.brain_shadow

    if word not in brain["spectral_semantics"]:
        return None

    return brain["spectral_semantics"][word]


# =====================================================
# CONVERSATION
# =====================================================

def send_message():

    text=st.session_state.user_input

    if not text.strip():
        return

    st.session_state.messages.append({

        "role":"user",
        "content":text
    })

    reply=respond(text)

    st.session_state.messages.append({

        "role":"assistant",
        "content":reply
    })

    st.session_state.brain_shadow=load_brain()

    st.session_state.user_input=""


# =====================================================
# RESET
# =====================================================

def reset():

    st.session_state.messages=[]

    st.rerun()


# =====================================================
# UI
# =====================================================

st.title("🧠 ORACLE V13 — Cognitive AI Laboratory")

brain=st.session_state.brain_shadow


# =====================================================
# METRICS
# =====================================================

c1,c2,c3,c4=st.columns(4)

c1.metric(
"Âge Cognitif",
brain["cortex"]["cognitive_age"]
)

c2.metric(
"Vocabulaire",
len(brain["lexical_memory"]["tokens"])
)

c3.metric(
"Densité Associative",
association_density()
)

c4.metric(
"Entropie Sémantique",
semantic_entropy()
)


st.info(diagnose())


# =====================================================
# WORKSPACE GLOBAL
# =====================================================

st.subheader("🧠 Global Workspace")

workspace=brain["global_workspace"]

col1,col2=st.columns(2)

with col1:

    st.write("Concepts actifs")

    st.write(workspace["active_concepts"])

with col2:

    st.write("Concept dominant")

    st.write(workspace["winning_concept"])


# =====================================================
# UPLOAD LEARNING
# =====================================================

st.subheader("📥 Nourrir l'IA")

file=st.file_uploader(
"Corpus cognitif",
type=["txt","csv","xlsx"]
)

if file:

    txt=read_file(file)

    respond(txt)

    st.session_state.brain_shadow=load_brain()

    st.success("Corpus assimilé")


# =====================================================
# CONVERSATION
# =====================================================

st.subheader("💬 Conversation")

col_reset,_=st.columns([1,5])

with col_reset:

    if st.button("Réinitialiser"):
        reset()


for msg in st.session_state.messages:

    if msg["role"]=="user":

        st.markdown(f"👤 **Vous :** {msg['content']}")

    else:

        st.markdown(f"🧠 **Oracle :** {msg['content']}")


st.text_input(
"Votre message",
key="user_input",
on_change=send_message
)


# =====================================================
# SPECTRAL SEMANTICS
# =====================================================

st.subheader("🔬 Analyse Spectrale")

word=st.text_input("Mot à analyser")

if word:

    spec=spectral_analysis(word)

    if spec:

        c1,c2=st.columns(2)

        c1.metric("Omega",spec["omega"])

        c2.metric("Alpha",spec["alpha"])

    else:

        st.warning("Aucune donnée spectrale pour ce mot.")


# =====================================================
# MEMORY INSPECTOR
# =====================================================

st.subheader("📚 Mémoire de l'IA")

with st.expander("Voir mémoire lexicale"):

    df=pd.DataFrame(brain["lexical_memory"]["tokens"]).T

    st.dataframe(df)


with st.expander("Voir graphe sémantique"):

    st.json(brain["semantic_graph"])


with st.expander("Voir mémoire conversationnelle"):

    st.json(brain["conversation_memory"])


# =====================================================
# EXPORT
# =====================================================

st.subheader("📤 Export cerveau IA")

if st.button("Exporter cerveau"):

    with open(BRAIN_FILE) as f:

        data=f.read()

    st.download_button(

        "Télécharger oracle_brain.json",

        data,

        file_name="oracle_v13_brain.json"
    )
