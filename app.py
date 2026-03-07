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
# INTERFACE
# ============================================================

question = st.text_input("Pose une question à ORACLE")

if question:

    response = oracle.reason(question)

    st.markdown("### Réponse")
    st.write(response)

# ============================================================
# AJOUT MEMOIRE
# ============================================================

st.markdown("---")
st.subheader("Ajouter un souvenir")

memory = st.text_input("Nouvelle information")

if st.button("Enregistrer"):
    oracle.add_memory(memory)
    st.success("Mémoire ajoutée")
