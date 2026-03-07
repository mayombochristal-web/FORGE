# ============================================================
# ORACLE V15 Ω COSMOS
# Interface Streamlit
# ============================================================

import streamlit as st
from oracle_engine_v15 import OracleEngine

oracle = OracleEngine()

st.set_page_config(
    page_title="ORACLE V15 Ω COSMOS",
    layout="wide"
)

st.title("🧠 ORACLE V15 Ω COSMOS")

st.markdown(
"""
Moteur cognitif expérimental

Architecture :
- Mémoire vectorielle
- Raisonnement trigram
- Analyse documentaire
- Apprentissage continu
"""
)

# ============================================================
# STAT
# ============================================================

st.metric("Souvenirs enregistrés", oracle.stats())

st.divider()

# ============================================================
# QUESTION
# ============================================================

st.subheader("💬 Dialogue avec ORACLE")

question = st.text_area("Pose ta question")

if st.button("Interroger ORACLE"):

    oracle.learn(question)

    response = oracle.reason(question)

    st.write(response)

# ============================================================
# DOCUMENT
# ============================================================

st.divider()

st.subheader("📚 Nourriture cérébrale")

uploaded = st.file_uploader(
    "Importer un fichier",
    type=["txt"]
)

if uploaded:

    text = uploaded.read().decode("utf-8")

    learned = oracle.learn_document(text)

    st.success(f"{learned} blocs de connaissance ajoutés")

# ============================================================
# RAPPORT
# ============================================================

st.divider()

st.subheader("📊 Analyse cognitive")

if st.button("Analyser mémoire"):

    report = oracle.report()

    st.write("Souvenirs :", report["souvenirs_totaux"])

    st.write("Sources :", report["sources"])

    st.write("Concepts dominants :", report["concepts"])