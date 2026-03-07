import streamlit as st
from oracle_engine import OracleEngine

st.set_page_config(
    page_title="ORACLE V15 Ω COSMOS",
    layout="wide"
)

st.title("🧠 ORACLE V15 Ω COSMOS")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle
• raisonnement multi-concept
• apprentissage documentaire
• mémoire persistante GitHub
• analyse cognitive
""")

if "oracle" not in st.session_state:

    st.session_state.oracle = OracleEngine()

oracle = st.session_state.oracle

st.metric("Souvenirs enregistrés", oracle.stats())

st.divider()

st.subheader("💬 Dialogue")

question = st.text_area("Pose une question")

if st.button("Interroger ORACLE"):

    response = oracle.reason(question)

    st.write(response)

st.divider()

st.subheader("📚 Apprentissage")

text = st.text_area("Ajouter connaissance")

if st.button("Apprendre"):

    blocks = oracle.learn(text)

    st.success(f"{blocks} souvenirs ajoutés")

st.divider()

st.subheader("📊 Analyse mémoire")

if st.button("Rapport mémoire"):

    r = oracle.report()

    st.write("Souvenirs :", r["souvenirs_totaux"])

    st.write("Concepts :", r["concepts"])
