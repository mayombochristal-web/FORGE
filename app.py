import streamlit as st
from oracle_engine import OracleEngine

st.set_page_config(page_title="ORACLE V18", layout="wide")

st.title("🧠 ORACLE V18 — Moteur cognitif")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle indexée  
• transformateur d'attention  
• raisonnement multi-concept  
• base de connaissances dynamique
""")


# ==========================================
# LOAD ORACLE (CACHE)
# ==========================================

@st.cache_resource
def load_oracle():

    return OracleEngine()

oracle = load_oracle()


# ==========================================
# QUESTION
# ==========================================

question = st.text_input("Pose une question à ORACLE")

if st.button("Interroger"):

    if question:

        response = oracle.generate_response(question)

        st.success(response)


# ==========================================
# FILE MEMORY
# ==========================================

st.subheader("Ajouter un document à la mémoire")

uploaded_file = st.file_uploader(
    "Importer un fichier",
    type=["txt","csv","pdf","docx"]
)

if uploaded_file:

    result = oracle.add_file_memory(uploaded_file)

    st.success(result)
