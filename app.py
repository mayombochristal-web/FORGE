import streamlit as st
import pandas as pd
import PyPDF2
import docx
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
    st.error("Erreur de configuration GitHub dans les Secrets.")
    st.stop()

# =====================================================
# 🛠️ LOGIQUE DE ROUTAGE ET SYNCHRONISATION
# =====================================================

def choisir_banques(message_utilisateur):
    """
    Analyse le message pour choisir :
    1. La banque principale (Base de réponse)
    2. La banque Nexus (Base de relation/comparaison)
    """
    msg = message_utilisateur.lower()
    
    # Mapping Mots-clés -> Fichiers
    mapping = {
        "technique.json": ["code", "python", "github", "api", "forge", "système"],
        "philosophie.json": ["vie", "sens", "oracle", "pensée", "exister", "humain"],
        "projets.json": ["travail", "client", "développement", "plan", "étape"],
        "personnel.json": ["moi", "famille", "goût", "humeur", "souvenir"]
    }

    principale = "oracle_memory.json"
    secondaire = None

    # Trouver la banque principale
    for fichier, mots in mapping.items():
        if any(m in msg for m in mots):
            principale = fichier
            break
    
    # Trouver une banque secondaire différente pour le Nexus
    for fichier, mots in mapping.items():
        if any(m in msg for m in mots) and fichier != principale:
            secondaire = fichier
            break
            
    return principale, secondaire

def charger_memoire_github(nom_fichier):
    if not nom_fichier: return False
    path = f"{FOLDER}/{nom_fichier}"
    try:
        content = repo.get_contents(path, ref=BRANCH)
        data = json.loads(content.decoded_content.decode("utf-8"))
        with open(nom_fichier, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

def sauvegarder_memoire_github(nom_fichier, commit_msg="Update Oracle"):
    if not os.path.exists(nom_fichier): return
    path = f"{FOLDER}/{nom_fichier}"
    with open(nom_fichier, "r", encoding="utf-8") as f:
        nouveau_contenu = f.read()
    try:
        contents = repo.get_contents(path, ref=BRANCH)
        repo.update_file(contents.path, commit_msg, nouveau_contenu, contents.sha, branch=BRANCH)
    except:
        repo.create_file(path, commit_msg, nouveau_contenu, branch=BRANCH)

# =====================================================
# 🚀 INITIALISATION DE L'INTERFACE
# =====================================================
st.set_page_config(page_title="ORACLE V6 — Cognition Unifiée", layout="wide")
st.title("🧠 ORACLE V6 — Intelligence Relationnelle")

# Initialisation session_state
if "current_mem" not in st.session_state:
    st.session_state.current_mem = "oracle_memory.json"
    charger_memoire_github(st.session_state.current_mem)
    st.session_state.brain = OracleBrain(st.session_state.current_mem)

brain = st.session_state.brain

# =====================================================
# 💬 ONGLET CONVERSATION (Avec Routage & Nexus)
# =====================================================
tab1, tab2, tab3 = st.tabs(["💬 Conversation", "📚 Nourrir", "🌐 GitHub Cloud"])

with tab1:
    # Affichage historique
    for msg in brain.dialog_memory:
        st.chat_message("assistant" if "Oracle:" in msg else "user").write(msg)

    user_msg = st.chat_input("Posez une question à l'Oracle...")

    if user_msg:
        # 1. Détection des banques de données
        banque_p, banque_s = choisir_banques(user_msg)
        
        with st.status("Réflexion cognitive...") as status:
            # Switch de banque principale si nécessaire
            if banque_p != st.session_state.current_mem:
                status.update(label=f"Chargement de la base : {banque_p}...")
                charger_memoire_github(banque_p)
                st.session_state.brain = OracleBrain(banque_p)
                st.session_state.current_mem = banque_p
            
            # Chargement du Nexus (Relation croisée)
            if banque_s:
                status.update(label=f"Création d'un lien avec : {banque_s}...")
                charger_memoire_github(banque_s)
                st.session_state.brain.cross_reference(banque_s)
            
            # Traitement
            response = st.session_state.brain.process_input(user_msg)
            
            # Sauvegarde automatique
            status.update(label="Synchronisation Cloud...")
            sauvegarder_memoire_github(banque_p, f"Sujet : {user_msg[:30]}")
            status.update(label="Terminé", state="complete")

        st.rerun()

# =====================================================
# 📚 ONGLET NOURRIR (Apprentissage)
# =====================================================
with tab2:
    st.subheader("Nourrir une banque spécifique")
    target_file = st.selectbox("Cible de l'apprentissage", ["oracle_memory.json", "technique.json", "philosophie.json", "projets.json"])
    text_input = st.text_area("Contenu à assimiler")
    
    if st.button("🧠 Assimiler dans le Cloud"):
        if text_input:
            charger_memoire_github(target_file)
            temp_brain = OracleBrain(target_file)
            temp_brain.process_input(text_input)
            sauvegarder_memoire_github(target_file, "Nouvelle assimilation")
            st.success(f"Apprentissage réussi dans {target_file}")

# =====================================================
# 🌐 ONGLET CLOUD (Gestion des fichiers)
# =====================================================
with tab3:
    st.subheader("Fichiers sur votre dépôt FORGE")
    try:
        items = repo.get_contents(FOLDER, ref=BRANCH)
        for item in items:
            col1, col2 = st.columns([4, 1])
            col1.code(item.path)
            if col2.button("Forcer Charge", key=item.name):
                charger_memoire_github(item.name)
                st.session_state.brain = OracleBrain(item.name)
                st.session_state.current_mem = item.name
                st.rerun()
    except:
        st.error("Dossier introuvable sur GitHub.")

# =====================================================
# 📊 SIDEBAR (État Interne)
# =====================================================
with st.sidebar:
    st.header("⚙️ État Interne")
    st.info(f"Base actuelle : **{st.session_state.current_mem}**")
    
    st.divider()
    st.subheader("Φ Dynamique")
    for k, v in brain.phi.items():
        st.progress(v, text=f"{k} : {v:.2f}")

    if st.button("🌙 Forcer Sommeil (Oubli)"):
        brain.sleep_cycle()
        sauvegarder_memoire_github(st.session_state.current_mem, "Sommeil profond")
        st.rerun()
