# ============================================================
# ORACLE V14 Ω TTU
# Interface Streamlit
# ============================================================

import streamlit as st
import os

from oracle_engine_v15 import OracleEngine
from ttu_file_scanner import scan_file
from analytics_engine import (
    export_csv,
    export_json,
    export_txt,
    analyze_memories,
    graph_progress,
    graph_sources
)
from github_memory import backup_memory

# ============================================================
# INITIALISATION
# ============================================================

oracle = OracleEngine()

st.set_page_config(page_title="ORACLE V14 Ω", layout="wide")

st.title("🧠 ORACLE V14 Ω TTU")

st.markdown("**Moteur cognitif expérimental**")

# ============================================================
# STATISTIQUES
# ============================================================

st.metric("Souvenirs", oracle.stats())

st.divider()

# ============================================================
# DIALOGUE
# ============================================================

st.subheader("Dialogue avec Oracle")

text = st.text_area("Pose ta question")

if st.button("Envoyer"):

    if text.strip() != "":

        oracle.learn(text)

        r = oracle.reason(text)

        st.success("Réponse ORACLE")

        st.write(r)

    else:

        st.warning("Veuillez écrire une question")

# ============================================================
# NOURRITURE CÉRÉBRALE
# ============================================================

st.divider()

st.subheader("Nourriture cérébrale")

uploaded = st.file_uploader(
    "Glissez-déposez un fichier",
    type=["txt","pdf","docx","csv","json"]
)

if uploaded:

    try:

        content = scan_file(uploaded)

        if content.strip() != "":

            oracle.learn(content, "document")

            st.success("Document analysé et appris")

        else:

            st.warning("Le fichier ne contient pas de texte exploitable")

    except Exception as e:

        st.error(f"Erreur analyse fichier : {e}")

# ============================================================
# EXPORT RAPPORTS
# ============================================================

st.divider()

st.subheader("Rapports IA")

if st.button("Générer rapports"):

    try:

        df = export_csv("oracle_memory/oracle.db")

        export_json(df)
        export_txt(df)

        st.success("Rapports générés")

        # CSV
        with open("oracle_report.csv","rb") as f:

            st.download_button(
                "Télécharger CSV",
                f,
                file_name="oracle_report.csv",
                mime="text/csv"
            )

        # JSON
        with open("oracle_report.json","rb") as f:

            st.download_button(
                "Télécharger JSON",
                f,
                file_name="oracle_report.json",
                mime="application/json"
            )

        # TXT
        with open("oracle_report.txt","rb") as f:

            st.download_button(
                "Télécharger TXT",
                f,
                file_name="oracle_report.txt",
                mime="text/plain"
            )

    except Exception as e:

        st.error(f"Erreur export : {e}")

# ============================================================
# ANALYSE DES SOUVENIRS
# ============================================================

st.divider()

st.subheader("Analyse cognitive")

if st.button("Analyser souvenirs"):

    stats = analyze_memories()

    st.write("🧠 Souvenirs totaux :", stats["total_memories"])

    st.write("📚 Sources :", stats["sources"])

    st.write("🔑 Concepts dominants :", stats["top_words"])

    img = graph_sources()

    if img and os.path.exists(img):

        st.image(img)

# ============================================================
# GRAPH PROGRESSION
# ============================================================

st.divider()

st.subheader("Progression IA")

if st.button("Graph évolution"):

    img = graph_progress("oracle_memory/oracle.db")

    if img and os.path.exists(img):

        st.image(img)

    else:

        st.warning("Pas assez de données pour générer le graphique")

# ============================================================
# BACKUP GITHUB
# ============================================================

st.divider()

st.subheader("Sauvegarde")

if st.button("Backup GitHub"):

    try:

        token = st.secrets["GITHUB_TOKEN"]

        repo = st.secrets["GITHUB_REPO"]

        backup_memory(token, repo, "oracle_memory")

        st.success("Backup GitHub effectué")

    except Exception as e:

        st.error(f"Erreur backup : {e}")