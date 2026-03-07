import streamlit as st
  from oracle_engine import OracleEngine

st.set_page_config(
    page_title="ORACLE V15.5 Ω COSMOS",
    layout="wide"
)

st.title("🧠 ORACLE V15.5 Ω COSMOS")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle  
• graph de connaissances  
• raisonnement multi-souvenirs  
• apprentissage documentaire  
• mémoire persistante GitHub  
• analyse cognitive  
""")

# INITIALISATION

if "oracle" not in st.session_state:

    st.session_state.oracle = OracleEngine()

oracle = st.session_state.oracle

st.metric("Souvenirs enregistrés", oracle.stats())

st.divider()

# QUESTION

st.subheader("💬 Dialogue avec ORACLE")

question = st.text_area("Pose une question")

if st.button("Interroger ORACLE"):

    response = oracle.reason(question)

    st.write(response)

# APPRENTISSAGE TEXTE

st.divider()

st.subheader("🧠 Ajouter connaissance")

text = st.text_area("Texte à apprendre")

if st.button("Apprendre texte"):

    blocks = oracle.learn(text)

    st.success(f"{blocks} souvenirs ajoutés")

# DOCUMENT

st.divider()

st.subheader("📚 Apprentissage documentaire")

uploaded = st.file_uploader(
    "Importer document",
    type=["txt","pdf","docx","csv"]
)

if uploaded:

    learned = oracle.learn_document(uploaded)

    st.success(f"{learned} blocs de connaissance ajoutés")

# ANALYSE

st.divider()

st.subheader("📊 Analyse mémoire")

if st.button("Analyser mémoire"):

    report = oracle.report()

    st.write("Souvenirs :", report["souvenirs_totaux"])

    st.write("Sources :", report["sources"])

    st.write("Concepts dominants :", report["concepts"])
