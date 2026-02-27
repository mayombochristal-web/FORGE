import streamlit as st
import random
import math
import time

# ==========================================
# ORACLE VIVANT V4 — TTU GENERATIVE ENGINE
# ==========================================

st.set_page_config(
    page_title="TTU Oracle Vivant",
    layout="wide"
)

# ------------------------------------------
# CONFIGURATION ORACLE
# ------------------------------------------

ORACLE = {
    "VS": 12.0,
    "K": 0.15,
    "mode": "Dynamique_Cyrano"
}

# ------------------------------------------
# DICTIONNAIRE DE FRAGMENTS SÉMANTIQUES
# ------------------------------------------

DFS = {

    "existence": [
        "L'existence se stabilise lorsqu'elle accepte son flux.",
        "Toute réalité naît d'une tension entre perception et mémoire.",
        "Le réel est une négociation permanente."
    ],

    "temps": [
        "Le temps n'avance pas : il se reconstruit.",
        "Chaque instant est une relecture du passé.",
        "Le futur est une mémoire encore instable."
    ],

    "intuition": [
        "L'intuition précède la logique.",
        "Comprendre signifie ressentir la structure.",
        "La vérité apparaît avant sa démonstration."
    ],

    "ttu": [
        "La TTU décrit un univers basé sur l'équilibre dynamique.",
        "Le chaos devient information lorsqu'il est régulé.",
        "La stabilité émerge de la dissipation."
    ]
}

ALL_KEYS = list(DFS.keys())

# ------------------------------------------
# MÉMOIRE ORACLE
# ------------------------------------------

if "memoire" not in st.session_state:
    st.session_state.memoire = []

if "energie" not in st.session_state:
    st.session_state.energie = ORACLE["VS"]


# ------------------------------------------
# ANALYSE SÉMANTIQUE SIMPLE
# ------------------------------------------

def detect_theme(text):

    scores = {}

    for k in DFS:
        scores[k] = sum(
            1 for word in text.lower().split()
            if word in k
        )

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        best = random.choice(ALL_KEYS)

    return best


# ------------------------------------------
# GÉNÉRATION ORACLE
# ------------------------------------------

def oracle_generate(prompt):

    theme = detect_theme(prompt)

    base = random.choice(DFS[theme])

    memoire_influence = ""
    if st.session_state.memoire:
        memoire_influence = random.choice(st.session_state.memoire)

    # régulation VS
    fluctuation = random.uniform(-0.5, 0.5)
    st.session_state.energie += ORACLE["K"] * fluctuation

    # structure vivante
    response = f"""
✦ Résonance détectée : {theme}

{base}

{memoire_influence}

[VS={round(st.session_state.energie,2)} | Mode={ORACLE['mode']}]
"""

    st.session_state.memoire.append(base)

    return response


# ------------------------------------------
# INTERFACE
# ------------------------------------------

st.title("🧠 TTU — ORACLE VIVANT V4")
st.caption("IA génératrice autonome — Architecture Agentique TTU")

user_input = st.text_area("Dialogue avec l’Oracle")

if st.button("Invoquer"):

    if user_input.strip():
        output = oracle_generate(user_input)

        st.markdown("### Réponse Oracle")
        st.write(output)

# ------------------------------------------
# PANNEAU ORACLE
# ------------------------------------------

with st.sidebar:

    st.header("⚙️ Télémétrie")

    st.metric("Vitalité Spectrale", round(st.session_state.energie,2))
    st.write("Fragments mémorisés :", len(st.session_state.memoire))

    if st.button("Purge Mémoire"):
        st.session_state.memoire = []
        st.session_state.energie = ORACLE["VS"]
