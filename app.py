import streamlit as st
from oracle_engine import OracleEngine

st.set_page_config(page_title="ORACLE V18", layout="wide")

st.title("🧠 ORACLE V18 — Moteur cognitif")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle indexée  
• transformateur sémantique  
• raisonnement multi-concept  
• base de connaissances dynamique
""")

# =========================================
# LOAD ORACLE (CACHE V18)
# =========================================

@st.cache_resource
def load_oracle():

    return OracleEngine()

oracle = load_oracle()


# =========================================
# QUESTION
# =========================================

st.subheader("Poser une question")

question = st.text_input("Pose une question à ORACLE")

if st.button("Interroger"):

    if question:

        response = oracle.reason(question)

        st.write(response)


# =========================================
# IMPORT DOCUMENT
# =========================================

st.subheader("Ajouter un document à la mémoire")

uploaded_file = st.file_uploader(
    "Importer un fichier",
    type=["txt","pdf","docx","csv","xlsx"]
)

if uploaded_file:

    with st.spinner("Apprentissage en cours..."):

        n = oracle.learn_document(uploaded_file)

    st.success(f"{n} blocs appris")


# =========================================
# STATS
# =========================================

st.subheader("Statistiques mémoire")

stats = oracle.stats()

st.write("Souvenirs :", stats["souvenirs"])
st.write("Sources :", stats["sources"])
