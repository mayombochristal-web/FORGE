import streamlit as st
from oracle_engine_v20 import OracleEngine

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="ORACLE V20",
    layout="wide"
)

# ============================================================
# INITIALISATION MOTEUR
# ============================================================

@st.cache_resource
def load_engine():
    return OracleEngine()

oracle = load_engine()

# ============================================================
# HEADER
# ============================================================

st.title("🧠 ORACLE V20")
st.caption("Moteur cognitif expérimental")

# ============================================================
# MENU
# ============================================================

menu = st.sidebar.selectbox(
    "Navigation",
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

if menu == "Accueil":

    st.subheader("Présentation")

    st.markdown(
"""
ORACLE est un moteur cognitif expérimental basé sur une mémoire linguistique hiérarchique.

Fonctionnalités :

• apprentissage de documents  
• mémoire vectorielle  
• graphe conceptuel  
• raisonnement linguistique  
• sauvegarde mémoire  

Architecture :

Document  
↓  
Paragraphe  
↓  
Phrase  
↓  
Mot  
↓  
Syllabe  
↓  
Caractère
"""
)

# ============================================================
# APPRENTISSAGE DOCUMENT
# ============================================================

elif menu == "Apprentissage document":

    st.subheader("Importer un document")

    file = st.file_uploader(
        "Choisir un fichier texte",
        type=["txt","md"]
    )

    if file:

        if st.button("Apprendre le document"):

            with st.spinner("Analyse et indexation..."):

                count = oracle.learn_document(file)

            st.success(f"{count} phrases apprises")

# ============================================================
# QUESTION ORACLE
# ============================================================

elif menu == "Question ORACLE":

    st.subheader("Poser une question")

    question = st.text_input("Question")

    if question:

        if st.button("Interroger ORACLE"):

            with st.spinner("Raisonnement en cours..."):

                answer = oracle.reason(question)

            st.markdown(answer)

# ============================================================
# STATISTIQUES
# ============================================================

elif menu == "Statistiques mémoire":

    st.subheader("Statistiques")

    stats = oracle.stats()

    col1,col2,col3 = st.columns(3)

    col1.metric("Documents",stats["documents"])
    col2.metric("Souvenirs (phrases)",stats["souvenirs"])
    col3.metric("Mots",stats["mots"])

# ============================================================
# EXPLORATION MEMOIRE
# ============================================================

elif menu == "Exploration mémoire":

    st.subheader("Recherche mémoire")

    query = st.text_input("Recherche phrase")

    if query:

        results = oracle.search_sentences(query)

        for r in results:

            st.write("Score:",round(r[0],3))
            st.write(r[1])
            st.divider()

# ============================================================
# ARCHITECTURE MEMOIRE
# ============================================================

elif menu == "Architecture mémoire":

    st.subheader("Architecture cognitive ORACLE")

    st.markdown(
"""
### Mémoire linguistique hiérarchique

La mémoire d'ORACLE est organisée selon plusieurs couches linguistiques :

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

Chaque couche est indexée séparément et reliée aux autres.

---

### Structure réelle de la mémoire

oracle_memory/

characters.db  
syllables.db  
words.db  
sentences.db  
paragraphs.db  
contexts.db  
documents.db  
concept_graph.db  

---

### Processus d'apprentissage

Lorsqu'un document est importé :

1. extraction texte  
2. découpage paragraphes  
3. découpage phrases  
4. extraction mots  
5. segmentation syllabes  
6. indexation caractères  

---

### Raisonnement ORACLE

Question  
↓  
Analyse linguistique  
↓  
Embedding sémantique  
↓  
Recherche phrases proches  
↓  
Scoring d'attention  
↓  
Croisement graphe conceptuel  
↓  
Génération de réponse  

---

### Sauvegarde mémoire

La mémoire est automatiquement sauvegardée sur GitHub.
"""
)