# ============================================================
# ORACLE V16 Ω COSMOS
# Interface Streamlit
# ============================================================

import streamlit as st
from oracle_engine_v16 import OracleEngine

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="ORACLE V16 Ω COSMOS",
    layout="wide"
)

st.title("🧠 ORACLE V16 Ω COSMOS")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle  
• raisonnement multi-concept  
• apprentissage documentaire  
• mémoire persistante compressée  
• graphe conceptuel dynamique  
• moteur d’attention cognitive  
• analyse cognitive avancée
""")

# ============================================================
# INITIALISATION ORACLE
# ============================================================

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleEngine()

oracle = st.session_state.oracle

# ============================================================
# STATISTIQUES
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.metric("Souvenirs enregistrés", oracle.stats())

with col2:
    report = oracle.report()
    st.metric("Concepts uniques", len(report["concepts"]))

st.divider()

# ============================================================
# DIALOGUE
# ============================================================

st.subheader("💬 Dialogue avec ORACLE")

question = st.text_area(
    "Pose ta question à ORACLE",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    ask = st.button("Interroger ORACLE")

with col2:
    learn = st.button("Apprendre ce texte")

if ask and question:

    # apprentissage du texte
    oracle.learn(question)

    # raisonnement
    response = oracle.reason(question)

    st.markdown("### 🧠 Réponse ORACLE")
    st.write(response)

if learn and question:

    oracle.learn(question)

    st.success("Texte appris par ORACLE")

# ============================================================
# INGESTION DOCUMENTAIRE
# ============================================================

st.divider()

st.subheader("📚 Nourriture cérébrale")

uploaded = st.file_uploader(
    "Importer un document",
    type=["txt","pdf","docx","csv"]
)

if uploaded:

    with st.spinner("Analyse et apprentissage du document..."):

        learned = oracle.learn_document(uploaded)

    st.success(f"{learned} blocs de connaissance ajoutés à la mémoire")

# ============================================================
# ANALYSE COGNITIVE
# ============================================================

st.divider()

st.subheader("📊 Analyse cognitive")

if st.button("Analyser mémoire ORACLE"):

    report = oracle.report()

    st.write("### Souvenirs totaux")
    st.write(report["souvenirs_totaux"])

    st.write("### Sources de connaissance")
    st.json(report["sources"])

    st.write("### Concepts dominants")
    st.json(report["concepts"])

# ============================================================
# GRAPHE CONCEPTUEL
# ============================================================

st.divider()

st.subheader("🧩 Concepts dominants")

if st.button("Afficher concepts principaux"):

    report = oracle.report()

    concepts = report["concepts"]

    if concepts:

        for c,v in concepts.items():

            st.write(f"{c} : {v}")

    else:

        st.info("Aucun concept dominant détecté.")

# ============================================================
# EXPORT MÉMOIRE
# ============================================================

st.divider()

st.subheader("💾 Sauvegarde mémoire")

if st.button("Sauvegarder mémoire ORACLE"):

    try:

        from github_memory import push_memory

        push_memory()

        st.success("Mémoire sauvegardée sur GitHub")

    except Exception as e:

        st.error(f"Erreur sauvegarde : {e}")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("ORACLE V16 Ω COSMOS — Architecture cognitive expérimentale")