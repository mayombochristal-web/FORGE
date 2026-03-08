import streamlit as st
from oracle_engine import OracleEngine

# =====================================================
# CONFIGURATION PAGE
# =====================================================

st.set_page_config(
    page_title="ORACLE V19",
    layout="wide"
)

st.title("🧠 ORACLE V19 — Moteur cognitif hiérarchique")

st.markdown("""
Architecture cognitive expérimentale

Capacités :

• mémoire linguistique hiérarchique  
• mémoire vectorielle indexée  
• transformateur sémantique simplifié  
• raisonnement multi-concepts  
• base de connaissances dynamique  
• sauvegarde mémoire automatique GitHub
""")

# =====================================================
# CHARGEMENT ORACLE
# =====================================================

@st.cache_resource
def load_oracle():

    return OracleEngine()

oracle = load_oracle()

# =====================================================
# MENU
# =====================================================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Interroger ORACLE",
        "Apprentissage document",
        "Statistiques mémoire",
        "Architecture mémoire"
    ]
)

# =====================================================
# QUESTION
# =====================================================

if menu == "Interroger ORACLE":

    st.subheader("Poser une question")

    question = st.text_input(
        "Pose une question à ORACLE"
    )

    if st.button("Interroger"):

        if question:

            with st.spinner("ORACLE réfléchit..."):

                response = oracle.reason(question)

            st.markdown("### Réponse")
            st.write(response)

# =====================================================
# APPRENTISSAGE DOCUMENT
# =====================================================

if menu == "Apprentissage document":

    st.subheader("Ajouter un document à la mémoire")

    uploaded_file = st.file_uploader(
        "Importer un fichier",
        type=["txt","pdf","docx","csv","xlsx"]
    )

    if uploaded_file:

        if st.button("Apprendre le document"):

            with st.spinner("Analyse linguistique en cours..."):

                n = oracle.learn_document(uploaded_file)

            st.success(f"{n} phrases apprises")

            st.info("""
            Le document a été analysé et intégré dans la mémoire hiérarchique :

            caractères → syllabes → mots → phrases → paragraphes → contextes
            """)

# =====================================================
# STATISTIQUES
# =====================================================

if menu == "Statistiques mémoire":

    st.subheader("Statistiques de la mémoire ORACLE")

    stats = oracle.stats()

    col1, col2 = st.columns(2)

    col1.metric(
        "Souvenirs (phrases)",
        stats["souvenirs"]
    )

    col2.metric(
        "Sources (documents)",
        stats["sources"]
    )

# =====================================================
# ARCHITECTURE
# =====================================================

if menu == "Architecture mémoire":

    st.subheader("Architecture cognitive ORACLE")

    st.markdown("""
### Mémoire linguistique hiérarchique

La mémoire d'ORACLE est organisée selon plusieurs couches linguistiques :