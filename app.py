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
# 🔐 CONFIGURATION GITHUB
# =====================================================
try:
    TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["GITHUB_REPO"]
    FOLDER = st.secrets["GITHUB_MEMORY_DIR"]
    BRANCH = st.secrets["GITHUB_BRANCH"]

    g = Github(TOKEN)
    repo = g.get_repo(REPO_NAME)
except Exception as e:
    st.error("⚠️ Configuration GitHub manquante dans les Secrets Streamlit.")
    st.stop()

# =====================================================
# 🛠️ FONCTIONS D'EXTRACTION & SYNCHRO
# =====================================================

def extract_multimodal(uploaded_file):
    """Extrait le contenu textuel de divers formats de fichiers."""
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join(p.extract_text() or "" for p in reader.pages)
        elif ext in ["docx", "doc"]:
            doc = docx.Document(uploaded_file)
            return " ".join(p.text for p in doc.paragraphs)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            return df.to_string()
        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        elif ext == "txt":
            return uploaded_file.read().decode("utf-8")
        elif ext in ["wav", "flac"]:
            r = sr.Recognizer()
            with sr.AudioFile(uploaded_file) as source:
                audio = r.record(source)
            return r.recognize_google(audio, language="fr-FR")
        return ""
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return ""

def charger_memoire_github(nom_fichier):
    path = f"{FOLDER}/{nom_fichier}"
    try:
        content = repo.get_contents(path, ref=BRANCH)
        data = json.loads(content.decoded_content.decode("utf-8"))
        with open(nom_fichier, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

def sauvegarder_memoire_github(nom_fichier, msg="Update"):
    if not os.path.exists(nom_fichier):
        return
    path = f"{FOLDER}/{nom_fichier}"
    with open(nom_fichier, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        git_file = repo.get_contents(path, ref=BRANCH)
        repo.update_file(git_file.path, msg, content, git_file.sha, branch=BRANCH)
    except:
        repo.create_file(path, msg, content, branch=BRANCH)

@st.cache_data(ttl=60)  # cache pendant 60 secondes
def lister_fichiers_json_github():
    """Retourne la liste des fichiers .json présents dans le dossier GitHub FOLDER."""
    try:
        contents = repo.get_contents(FOLDER, ref=BRANCH)
        fichiers = [c.name for c in contents if c.name.endswith('.json')]
        return fichiers
    except Exception as e:
        st.error(f"Impossible de lister les fichiers GitHub : {e}")
        return []

# =====================================================
# 🚀 INITIALISATION DE LA SESSION
# =====================================================
st.set_page_config(page_title="ORACLE V6", layout="wide", page_icon="🧠")

if "current_mem" not in st.session_state:
    st.session_state.current_mem = "oracle_memory.json"
    charger_memoire_github(st.session_state.current_mem)
    st.session_state.brain = OracleBrain(st.session_state.current_mem)

brain = st.session_state.brain

# =====================================================
# 🎨 INTERFACE UTILISATEUR (ONGLETS)
# =====================================================
tab1, tab2, tab3 = st.tabs(["💬 Conversation", "📚 Nourrir (Multimodal)", "🌐 Cloud & Recherche"])

# --- ONGLET 1 : CONVERSATION ---
with tab1:
    col_h, col_r = st.columns([4, 1])
    col_h.subheader(f"🧠 Session active : {st.session_state.current_mem}")

    if col_r.button("🔄 Réactiver le fil (Reset)"):
        brain.dialog_memory.clear()
        st.rerun()

    # Affichage des messages
    for i, msg in enumerate(brain.dialog_memory):
        is_oracle = "Oracle:" in msg
        with st.chat_message("assistant" if is_oracle else "user", avatar="🧠" if is_oracle else "👤"):
            clean_text = msg.replace("Oracle:", "").replace("User:", "").strip()
            st.write(clean_text)
            if is_oracle:
                st.button("📋 Copier la réponse", key=f"copy_{i}",
                         on_click=lambda t=clean_text: st.toast(f"Copié : {t[:50]}..."))

    user_msg = st.chat_input("Échangez avec l'Oracle...")
    if user_msg:
        with st.spinner("L'Oracle analyse..."):
            response = brain.process_input(user_msg)
            sauvegarder_memoire_github(st.session_state.current_mem, f"Échange : {user_msg[:20]}")
        st.rerun()

# --- ONGLET 2 : NOURRIR (EXTRACTION COMPLÈTE) ---
with tab2:
    st.subheader("📥 Importer des connaissances")

    # Récupération dynamique de la liste des fichiers JSON
    fichiers_json = lister_fichiers_json_github()
    if not fichiers_json:
        st.warning("Aucun fichier JSON trouvé dans le dossier GitHub. Utilisation de la liste par défaut.")
        fichiers_json = ["oracle_memory.json", "technique.json", "philosophie.json", "projets.json"]

    target_db = st.selectbox("Sélectionner la base de destination", fichiers_json)

    media = st.radio(
        "Format de la source",
        ["Document (PDF, Word, Excel, CSV, TXT)", "Audio (WAV, FLAC)", "Saisie Manuelle"]
    )

    input_data = ""
    if media == "Saisie Manuelle":
        input_data = st.text_area("Collez votre texte ici")
    elif media == "Audio (WAV, FLAC)":
        f = st.file_uploader("Fichier Audio", type=["wav", "flac"])
        if f:
            input_data = extract_multimodal(f)
    else:
        f = st.file_uploader("Fichier Document", type=["pdf", "docx", "xlsx", "xls", "csv", "txt"])
        if f:
            input_data = extract_multimodal(f)

    if st.button("🧠 Assimiler & Synchroniser"):
        if input_data:
            with st.spinner("Assimilation en cours..."):
                charger_memoire_github(target_db)
                temp_brain = OracleBrain(target_db)
                temp_brain.process_input(input_data)
                sauvegarder_memoire_github(target_db, "Assimilation automatique")
                st.success(f"Données intégrées avec succès dans {target_db} !")
        else:
            st.warning("Veuillez fournir un contenu.")

# --- ONGLET 3 : CLOUD & RECHERCHE ---
with tab3:
    st.subheader("📂 Gestionnaire de Mémoires Cloud")

    search_query = st.text_input("🔍 Rechercher un fichier mémoire (ex: 'technique', '2024'...)")

    try:
        all_items = repo.get_contents(FOLDER, ref=BRANCH)
        filtered_items = [i for i in all_items if search_query.lower() in i.name.lower()]

        if not filtered_items:
            st.info("Aucun fichier ne correspond à votre recherche.")

        for item in filtered_items:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"📄 `{item.name}`")
            if c2.button("Charger", key=f"load_{item.name}"):
                st.session_state.current_mem = item.name
                charger_memoire_github(item.name)
                st.session_state.brain = OracleBrain(item.name)
                st.rerun()
            if c3.button("🗑️", key=f"del_{item.name}", help="Supprimer du Cloud"):
                repo.delete_file(item.path, f"Suppression de {item.name}", item.sha, branch=BRANCH)
                st.rerun()
    except Exception as e:
        st.error(f"Accès au dépôt GitHub impossible : {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 État de l'Oracle")
    st.write(f"Base : **{st.session_state.current_mem}**")

    for k, v in brain.phi.items():
        st.progress(v, text=f"{k} : {v:.2f}")

    st.divider()
    if st.button("🌙 Sommeil Profond"):
        brain.sleep_cycle()
        sauvegarder_memoire_github(st.session_state.current_mem, "Cycle de sommeil")
        st.success("Mémoire nettoyée.")
        st.rerun()