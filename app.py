import streamlit as st
import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr
import json
import os
import time
from github import Github
from oracle_core import OracleBrain

# =====================================================
# 🔐 CONFIGURATION GITHUB (Via Secrets)
# =====================================================
try:
    TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["GITHUB_REPO"]
    FOLDER = st.secrets["GITHUB_MEMORY_DIR"]
    BRANCH = st.secrets["GITHUB_BRANCH"]
    
    g = Github(TOKEN)
    repo = g.get_repo(REPO_NAME)
except Exception as e:
    st.error("Erreur de configuration des Secrets GitHub. Vérifiez votre tableau de bord Streamlit.")
    st.stop()

# =====================================================
# 🧠 LOGIQUE DE SYNCHRONISATION GITHUB
# =====================================================

def charger_memoire_github(nom_fichier):
    """Télécharge le contenu JSON depuis GitHub vers le dossier local temporaire."""
    path = f"{FOLDER}/{nom_fichier}"
    try:
        content = repo.get_contents(path, ref=BRANCH)
        data = json.loads(content.decoded_content.decode("utf-8"))
        # On sauvegarde localement pour que OracleBrain puisse travailler
        with open(nom_fichier, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        # Si le fichier n'existe pas encore sur GitHub, on laisse Oracle en créer un nouveau
        return False

def sauvegarder_memoire_github(nom_fichier, commit_msg="Update Oracle Memory"):
    """Envoie le fichier local mis à jour vers GitHub."""
    path = f"{FOLDER}/{nom_fichier}"
    with open(nom_fichier, "r", encoding="utf-8") as f:
        nouveau_contenu = f.read()
    
    try:
        contents = repo.get_contents(path, ref=BRANCH)
        repo.update_file(contents.path, commit_msg, nouveau_contenu, contents.sha, branch=BRANCH)
    except:
        repo.create_file(path, commit_msg, nouveau_contenu, branch=BRANCH)

# =====================================================
# 🚀 INITIALISATION DE L'ORACLE
# =====================================================
st.set_page_config(page_title="ORACLE V6", page_icon="🧠", layout="wide")

# Choix du fichier de mémoire (Multi-fichiers)
st.sidebar.title("📁 Sélection de la Mémoire")
nom_memoire = st.sidebar.text_input("Fichier actif", "oracle_memory.json")

if "brain" not in st.session_state or st.session_state.get('current_mem') != nom_memoire:
    with st.spinner("Synchronisation avec GitHub..."):
        charger_memoire_github(nom_memoire)
        st.session_state.brain = OracleBrain(nom_memoire)
        st.session_state.current_mem = nom_memoire

brain = st.session_state.brain

# =====================================================
# 📂 FONCTIONS D'EXTRACTION (Multimodal)
# =====================================================
def extract_from_file(uploaded_file):
    if uploaded_file is None: return ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join(p.extract_text() or "" for p in reader.pages)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return " ".join(p.text for p in doc.paragraphs)
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        # Ajoutez ici les autres formats (csv, audio...)
        return ""
    except Exception as e:
        st.error(f"Erreur : {e}")
        return ""

# =====================================================
# 🎨 INTERFACE UTILISATEUR
# =====================================================
st.title("🧠 ORACLE V6 — Cognition Connectée")

tab1, tab2, tab3 = st.tabs(["💬 Conversation", "📚 Nourrir", "🌐 GitHub Cloud"])

with tab1:
    for msg in brain.dialog_memory:
        st.write(msg)
    
    user_msg = st.chat_input("Échanger avec l'Oracle...")
    if user_msg:
        with st.spinner("L'Oracle réfléchit..."):
            response = brain.process_input(user_msg, is_user=True)
            # SAUVEGARDE AUTOMATIQUE VERS GITHUB APRÈS CHAQUE RÉPONSE
            sauvegarder_memoire_github(nom_memoire, f"Conversation : {user_msg[:20]}...")
        st.rerun()

with tab2:
    mode = st.radio("Source", ["Texte", "Document"])
    content = ""
    if mode == "Texte":
        content = st.text_area("Collez votre texte")
    else:
        file = st.file_uploader("Fichier", type=["pdf", "docx", "txt"])
        if file: content = extract_from_file(file)

    if st.button("🧠 Assimiler"):
        if content:
            brain.process_input(content, is_user=True)
            sauvegarder_memoire_github(nom_memoire, "Assimilation document")
            st.success("Mémoire apprise et synchronisée sur GitHub !")
            st.rerun()

with tab3:
    st.subheader("Fichiers dans `oracle_memory/` sur GitHub")
    try:
        items = repo.get_contents(FOLDER, ref=BRANCH)
        for item in items:
            col_f, col_b = st.columns([3, 1])
            col_f.code(item.path)
            if col_b.button("Charger", key=item.path):
                nom_memoire = item.name
                st.rerun()
    except:
        st.write("Dossier vide ou inaccessible.")

# ---------- SIDEBAR ÉTAT ----------
with st.sidebar:
    st.divider()
    st.subheader("📊 État Cognitif")
    for k, v in brain.phi.items():
        st.progress(v, text=f"{k}")
    
    if st.button("💾 Forcer Synchro GitHub"):
        sauvegarder_memoire_github(nom_memoire, "Sauvegarde manuelle")
        st.toast("Mémoire envoyée sur GitHub !")
