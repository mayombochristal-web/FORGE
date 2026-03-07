import streamlit as st
from oracle_engine import OracleEngine

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="ORACLE V16",
    layout="wide"
)

st.title("🧠 ORACLE V16 — Cognitive Engine")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire vectorielle indexée  
• attention transformer  
• raisonnement multi-concept  
• base de connaissances dynamique  
""")

# ============================================================
# CACHE ORACLE (DEMARRAGE RAPIDE)
# ============================================================

@st.cache_resource
def load_oracle():
    return OracleEngine()

oracle = load_oracle()

# ============================================================
# QUESTION UTILISATEUR
# ============================================================

st.subheader("Poser une question")

question = st.text_input("Pose une question à ORACLE")

if question:

    response = oracle.reason(question)

    st.markdown("### Réponse")
    st.write(response)

# ============================================================
# AJOUT DOCUMENT
# ============================================================

st.markdown("---")
st.subheader("Ajouter un document à la mémoire")

uploaded_file = st.file_uploader(
    "Importer un fichier",
    type=["pdf", "docx", "xlsx", "csv", "txt"]
)

if uploaded_file:

    result = oracle.add_file_memory(uploaded_file)

    st.success(result)

# ============================================================
# AJOUT MEMOIRE TEXTE
# ============================================================

st.markdown("---")
st.subheader("Ajouter un souvenir")

memory = st.text_input("Nouvelle information")

if st.button("Enregistrer"):

    if memory.strip() != "":
        oracle.add_memory(memory)
        st.success("Mémoire ajoutée")
    else:
        st.warning("Veuillez entrer une information.")
