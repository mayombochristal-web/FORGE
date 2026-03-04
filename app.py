import streamlit as st
import time
import base64
import requests
import PyPDF2
import docx
import pandas as pd
import json
from oracle_core import OracleBrain

# --- CONFIGURATION GITHUB & FICHIERS ---
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
MEM_FILE = "oracle_memory.json"

def github_sync():
    """Synchronise la mémoire locale avec le dépôt GitHub distant."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        with open(MEM_FILE, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        # Récupération du SHA pour la mise à jour
        r = requests.get(url, headers=headers, timeout=5)
        sha = r.json()["sha"] if r.status_code == 200 else None
        
        data = {
            "message": "🧬 Oracle V4.5 Ω - Neural Sync",
            "content": content,
            "branch": "main"
        }
        if sha:
            data["sha"] = sha
            
        requests.put(url, headers=headers, json=data, timeout=10)
    except Exception as e:
        st.sidebar.warning(f"Sync GitHub différé : {e}")

# --- INITIALISATION DE L'ORACLE ---
if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain(MEM_FILE)
    if "chat" not in st.session_state:
        st.session_state.chat = []

st.set_page_config(page_title="ORACLE V4.5 Ω", page_icon="🧠", layout="wide")
st.title("🧠 ORACLE V4.5 Ω — Agent Cognitif")

# --- BARRE LATÉRALE (PARAMÈTRES & PHI) ---
with st.sidebar:
    st.header("⚙️ État Neuronal")
    
    # Affichage dynamique des constantes Phi
    for k, v in st.session_state.oracle.phi.items():
        label = k.replace("phi_", "Niveau ").upper()
        st.progress(v, text=f"{label}: {v:.2f}")
    
    st.markdown("---")
    
    if st.button("🌙 Sommeil Profond (Save)"):
        with st.spinner("Consolidation..."):
            st.session_state.oracle.save_all()
            github_sync()
            st.success("Mémoire sauvegardée et synchronisée.")

# --- INJECTION DE DONNÉES (NOURRIR L'ORACLE) ---
st.subheader("📥 Injection de Connaissances")
uploaded_file = st.file_uploader("Nourrir l'Oracle (PDF, TXT, DOCX, JSON)", type=["pdf", "txt", "docx", "json"])

if uploaded_file:
    with st.spinner("Lecture et fragmentation des données..."):
        content = ""
        try:
            if uploaded_file.type == "application/json":
                data = json.load(uploaded_file)
                content = json.dumps(data, ensure_ascii=False)
            elif uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            else:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
            
            if content and st.button("Confirmer l'Injection"):
                st.session_state.oracle.add_to_memory(content)
                st.success(f"L'Oracle a assimilé {len(content)} caractères.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture : {e}")

st.markdown("---")

# --- INTERFACE DE CONVERSATION ---
# Affichage de l'historique
for message in st.session_state.chat:
    with st.chat_message("assistant", avatar="🧠"):
        placeholder = st.empty()  # <--- Bien indenté (4 espaces)
        with st.spinner("L'Oracle analyse..."): # <--- Bien indenté (4 espaces)
            # Tout ce qui suit est indenté de 8 espaces car c'est dans le spinner
            response = st.session_state.oracle.generate_response(prompt)
            placeholder.markdown(response)

# Zone de saisie
if prompt := st.chat_input("Interroger l'Oracle..."):
    # 1. Message Utilisateur
    st.session_state.chat.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Réponse de l'Oracle
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Analyse neuronale en cours..."):
            # Appel du moteur core
            response = st.session_state.oracle.generate_response(prompt)
            
            # Affichage de la réponse
            st.markdown(response)
            
            # Zone de copie si le texte est volumineux
            if len(response) > 300:
                with st.expander("📄 Copier le texte intégral"):
                    st.code(response, language=None)
    
    # Sauvegarde dans l'historique de session
    st.session_state.chat.append({"role": "assistant", "content": response, "avatar": "🧠"})
    
    # Sauvegarde automatique en arrière-plan
    st.session_state.oracle.save_all()
    # Note: Sync GitHub peut être lent, on le fait ici ou via le bouton Sleep
