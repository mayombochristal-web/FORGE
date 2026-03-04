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
        
        r = requests.get(url, headers=headers, timeout=5)
        sha = r.json()["sha"] if r.status_code == 200 else None
        
        data = {"message": "🧬 Oracle Sync", "content": content, "branch": "main"}
        if sha: data["sha"] = sha
            
        requests.put(url, headers=headers, json=data, timeout=10)
    except Exception as e:
        st.sidebar.warning(f"Sync GitHub différé")

# --- INITIALISATION ---
if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain(MEM_FILE)
    if "chat" not in st.session_state:
        st.session_state.chat = []

st.set_page_config(page_title="ORACLE V4.5 Ω", page_icon="🧠", layout="wide")
st.title("🧠 ORACLE V4.5 Ω")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ État Neuronal")
    # Utilisation de colonnes pour stabiliser l'affichage des Phis
    cols = st.columns(3)
    for i, (k, v) in enumerate(st.session_state.oracle.phi.items()):
        cols[i % 3].metric(k.split('_')[1].upper(), f"{v:.2f}")
    
    st.markdown("---")
    if st.button("🌙 Sommeil Profond"):
        with st.spinner("Consolidation..."):
            st.session_state.oracle.save_all()
            github_sync()
            st.success("Mémoire sauvegardée.")

# --- INJECTION ---
st.subheader("📥 Injection de Connaissances")
uploaded_file = st.file_uploader("Nourrir l'Oracle", type=["pdf", "txt", "docx", "json"])

if uploaded_file:
    if st.button("Confirmer l'Injection"):
        with st.spinner("Assimilation..."):
            content = ""
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif uploaded_file.type == "application/json":
                content = json.dumps(json.load(uploaded_file))
            else:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
            
            st.session_state.oracle.add_to_memory(content)
            st.success("Données assimilées.")

st.markdown("---")

# --- CHAT INTERFACE ---
# Affichage stable de l'historique
for message in st.session_state.chat:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Interroger l'Oracle..."):
    st.session_state.chat.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        placeholder = st.empty() # Espace stable pour la réponse
        with st.spinner("Réflexion..."):
            response = st.session_state.oracle.generate_response(prompt)
            placeholder.markdown(response)
            
            if len(response) > 300:
                with st.expander("📄 Copier"):
                    st.code(response, language=None)
    
    st.session_state.chat.append({"role": "assistant", "content": response, "avatar": "🧠"})
    st.session_state.oracle.save_all()
