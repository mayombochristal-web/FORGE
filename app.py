# =====================================================
# ORACLE V14 Ω INTERFACE
# =====================================================

import streamlit as st
import pandas as pd
import json
from oracle_engine_v14 import OracleBrain

st.set_page_config(
    page_title="ORACLE Ω V14",
    layout="wide",
    page_icon="🧠"
)

# =====================================================
# INITIALISATION
# =====================================================

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain()

oracle = st.session_state.oracle

# =====================================================
# SIDEBAR — ETAT TTU
# =====================================================

with st.sidebar:

    st.title("TTU Cognitive State")

    phi = oracle.phi

    st.metric("Φ mémoire", f"{phi['phi_m']:.3f}")
    st.progress(phi["phi_m"])

    st.metric("Φ cohérence", f"{phi['phi_c']:.3f}")
    st.progress(phi["phi_c"])

    st.metric("Φ dissipation", f"{phi['phi_d']:.3f}")
    st.progress(phi["phi_d"])

    st.divider()

    st.metric("Distance attracteur", f"{oracle.distance():.4f}")

    st.metric("Concepts", oracle.memory_size())

    st.metric("Age", oracle.age)

    st.divider()

    if st.button("🌙 Sleep cycle"):
        oracle.sleep_cycle()
        st.success("Consolidation mémoire")

# =====================================================
# DIALOGUE
# =====================================================

st.header("Dialogue")

user = st.text_input("Parlez à l'Oracle")

if user:

    oracle.learn(user)

    answer = oracle.think(user)

    st.write("### Oracle")
    st.write(answer)

# =====================================================
# ANALYSE DOCUMENT
# =====================================================

st.header("Analyse de documents")

files = st.file_uploader(
    "Importer fichiers",
    accept_multiple_files=True,
    type=["pdf","docx","txt","csv","json","xlsx"]
)

if files:

    for file in files:

        text = oracle.read_document(file)

        if text:

            stats = oracle.analyze_document(text)

            st.success(file.name)

            st.write(stats)

            response = oracle.think(text)

            st.write("Réponse Oracle")
            st.write(response)

# =====================================================
# VISUALISATION CONCEPTUELLE
# =====================================================

st.header("Carte Conceptuelle")

if st.button("Afficher carte"):

    fig = oracle.visualize()

    if fig:
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EXPORT DONNEES
# =====================================================

st.header("Export")

if st.button("Exporter concepts"):

    df = oracle.export_concepts()

    st.download_button(
        "Télécharger CSV",
        df.to_csv(index=False),
        "oracle_concepts.csv"
    )

    st.download_button(
        "Télécharger JSON",
        df.to_json(orient="records"),
        "oracle_concepts.json"
    )