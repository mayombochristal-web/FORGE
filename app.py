# ============================================================
# ORACLE V15 Ω COSMOS
# Interface Streamlit
# Evolution stable de ORACLE V14 Ω TTU
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

st.set_page_config(
    page_title="ORACLE V15 Ω COSMOS",
    layout="wide"
)

st.title("🧠 ORACLE V15 Ω COSMOS")

st.markdown(
"""
**Moteur cognitif expérimental**

Architecture :
- Mémoire vectorielle
- Raisonnement trigram
- Analyse documentaire
- Apprentissage continu
- Rapports analytiques
"""
)

# ============================================================
# STATISTIQUES
# ============================================================

try:

    memories_count = oracle.stats()

except:

    memories_count = 0

st.metric("Souvenirs enregistrés", memories_count)

st.divider()

# ============================================================
# DIALOGUE ORACLE
# ============================================================

st.subheader("💬 Dialogue avec ORACLE")

text = st.text_area(
    "Pose ta question",
    height=120
)

if st.button("Interroger ORACLE"):

    if text.strip() != "":

        try:

            # apprentissage question
            oracle.learn(text)

            # raisonnement
            response = oracle.reason(text)

            st.success("Réponse générée")

            st.write(response)

        except Exception as e:

            st.error(f"Erreur moteur ORACLE : {e}")

    else:

        st.warning("Veuillez écrire une question.")

# ============================================================
# NOURRITURE CÉRÉBRALE
# ============================================================

st.divider()

st.subheader("📚 Nourriture cérébrale")

uploaded = st.file_uploader(
    "Importer un fichier pour nourrir l'ORACLE",
    type=["txt", "pdf", "docx", "csv", "json"]
)

if uploaded:

    try:

        content = scan_file(uploaded)

        if content.strip() != "":

            oracle.learn_document(content)

            st.success("Document analysé et mémorisé")

        else:

            st.warning("Le fichier ne contient pas de texte exploitable")

    except Exception as e:

        st.error(f"Erreur analyse fichier : {e}")

# ============================================================
# RAPPORTS IA
# ============================================================

st.divider()

st.subheader("📊 Rapports IA")

if st.button("Générer rapports analytiques"):

    try:

        df = export_csv("oracle_memory/oracle_core.db")

        export_json(df)
        export_txt(df)

        st.success("Rapports générés")

        # CSV
        with open("oracle_report.csv", "rb") as f:

            st.download_button(
                label="Télécharger CSV",
                data=f,
                file_name="oracle_report.csv",
                mime="text/csv"
            )

        # JSON
        with open("oracle_report.json", "rb") as f:

            st.download_button(
                label="Télécharger JSON",
                data=f,
                file_name="oracle_report.json",
                mime="application/json"
            )

        # TXT
        with open("oracle_report.txt", "rb") as f:

            st.download_button(
                label="Télécharger TXT",
                data=f,
                file_name="oracle_report.txt",
                mime="text/plain"
            )

    except Exception as e:

        st.error(f"Erreur export rapports : {e}")

# ============================================================
# ANALYSE COGNITIVE
# ============================================================

st.divider()

st.subheader("🧠 Analyse cognitive")

if st.button("Analyser la mémoire ORACLE"):

    try:

        stats = analyze_memories()

        st.write("🧠 Souvenirs totaux :", stats["total_memories"])

        st.write("📚 Sources :", stats["sources"])

        st.write("🔑 Concepts dominants :", stats["top_words"])

        img = graph_sources()

        if img and os.path.exists(img):

            st.image(img)

    except Exception as e:

        st.error(f"Erreur analyse : {e}")

# ============================================================
# PROGRESSION IA
# ============================================================

st.divider()

st.subheader("📈 Progression cognitive")

if st.button("Afficher évolution IA"):

    try:

        img = graph_progress("oracle_memory/oracle_core.db")

        if img and os.path.exists(img):

            st.image(img)

        else:

            st.warning("Pas assez de données pour générer le graphique")

    except Exception as e:

        st.error(f"Erreur graphique : {e}")

# ============================================================
# BACKUP GITHUB
# ============================================================

st.divider()

st.subheader("☁️ Sauvegarde GitHub")

if st.button("Sauvegarder mémoire ORACLE"):

    try:

        token = st.secrets["GITHUB_TOKEN"]

        repo = st.secrets["GITHUB_REPO"]

        backup_memory(token, repo, "oracle_memory")

        st.success("Backup GitHub effectué")

    except Exception as e:

        st.error(f"Erreur backup : {e}")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("ORACLE V15 Ω COSMOS — Architecture cognitive expérimentale")