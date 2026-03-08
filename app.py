import streamlit as st
from oracle_engine import OracleEngine

st.set_page_config(page_title="ORACLE V20.1",layout="wide")

@st.cache_resource
def load_engine():
    return OracleEngine()

oracle=load_engine()

st.title("🧠 ORACLE V20.1")

menu=st.sidebar.selectbox(
"Menu",
[
"Accueil",
"Apprentissage document",
"Question ORACLE",
"Statistiques mémoire",
"Exploration mémoire",
"Architecture mémoire"
]
)

# ============================================================
# ACCUEIL
# ============================================================

if menu=="Accueil":

    st.markdown("""
ORACLE V20.1 est un moteur cognitif expérimental.

Fonctionnalités :

• mémoire linguistique hiérarchique  
• vector search  
• graphe conceptuel  
• raisonnement linguistique  
• migration automatique V19  
""")

# ============================================================
# LEARNING
# ============================================================

elif menu=="Apprentissage document":

    file=st.file_uploader("Importer document",type=["txt","md"])

    if file:

        if st.button("Apprendre"):

            with st.spinner("Analyse en cours..."):

                count=oracle.learn_document(file)

            st.success(f"{count} phrases apprises")

# ============================================================
# QUESTION
# ============================================================

elif menu=="Question ORACLE":

    question=st.text_input("Question")

    if question:

        if st.button("Interroger"):

            answer=oracle.reason(question)

            st.markdown(answer)

# ============================================================
# STATS
# ============================================================

elif menu=="Statistiques mémoire":

    stats=oracle.stats()

    col1,col2,col3=st.columns(3)

    col1.metric("Documents",stats["documents"])
    col2.metric("Phrases",stats["souvenirs"])
    col3.metric("Mots",stats["mots"])

# ============================================================
# SEARCH
# ============================================================

elif menu=="Exploration mémoire":

    query=st.text_input("Recherche")

    if query:

        results=oracle.search_sentences(query)

        for r in results:

            st.write("Score:",round(r[0],3))
            st.write(r[1])
            st.divider()

# ============================================================
# ARCHITECTURE
# ============================================================

elif menu=="Architecture mémoire":

    st.markdown("""
### Architecture mémoire ORACLE

Document  
↓  
Contexte  
↓  
Paragraphes  
↓  
Phrases  
↓  
Mots  
↓  
Syllabes  
↓  
Caractères

### Bases de données

oracle_memory/

characters.db  
syllables.db  
words.db  
sentences.db  
paragraphs.db  
contexts.db  
documents.db  
concept_graph.db

### Pipeline cognitif

Question  
↓  
Embedding  
↓  
Recherche vectorielle  
↓  
Croisement graphe conceptuel  
↓  
Génération réponse
""")