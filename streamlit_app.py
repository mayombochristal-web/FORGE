import streamlit as st
from oracle_engine import OracleEngine

# Moteur Oracle unique
@st.cache_resource
def get_engine():
    return OracleEngine()

engine = get_engine()

st.title("Oracle Memory Engine")

st.sidebar.header("Actions")
mode = st.sidebar.radio("Choisir une action", ["Stats", "Apprendre un texte", "Uploader un document", "Poser une question"])

if mode == "Stats":
    stats = engine.stats()
    st.subheader("Statistiques")
    st.json(stats)

elif mode == "Apprendre un texte":
    st.subheader("Apprendre un texte")
    text = st.text_area("Texte à apprendre")
    if st.button("Apprendre"):
        if text.strip():
            nb_blocks = engine.learn(text, source="streamlit_form")
            st.success(f"{nb_blocks} bloc(s) appris avec succès")
        else:
            st.error("Aucun texte fourni")

elif mode == "Uploader un document":
    st.subheader("Uploader un document")
    file = st.file_uploader("Choisissez un fichier", type=["txt", "pdf", "docx", "csv", "xls", "xlsx"])
    if file is not None:
        if st.button("Apprendre ce document"):
            # streamlit donne un UploadedFile; on le passe directement à learn_document
            file.name = file.name  # déjà présent
            file.type = file.type  # déjà présent
            nb_blocks = engine.learn_document(file)
            st.success(f"{nb_blocks} bloc(s) appris depuis le fichier {file.name}")

elif mode == "Poser une question":
    st.subheader("Posez une question")
    question = st.text_input("Question")
    if st.button("Questionner"):
        if question.strip():
            answer = engine.reason(question)
            st.write("**Réponse :**")
            st.write(answer)
        else:
            st.error("Aucune question fournie")