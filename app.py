import streamlit as st
import time
import base64
import requests
import PyPDF2
import docx
import pandas as pd
from oracle_core import OracleBrain

# --- CONFIGURATION GITHUB ---
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
MEM_FILE = "oracle_memory.json"

def github_sync():
    if not GITHUB_TOKEN or not GITHUB_REPO: return
    try:
        with open(MEM_FILE, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        r = requests.get(url, headers=headers)
        sha = r.json()["sha"] if r.status_code == 200 else None
        data = {"message": "🧬 Oracle Sync", "content": content, "branch": "main"}
        if sha: data["sha"] = sha
        requests.put(url, headers=headers, json=data)
    except Exception as e: st.sidebar.error(f"GitHub Sync Error: {e}")

# --- INITIALISATION ---
if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain(MEM_FILE)
    if "chat" not in st.session_state: st.session_state.chat = []

st.set_page_config(page_title="ORACLE V4.5 Ω", page_icon="🧠", layout="wide")
st.title("🧠 ORACLE V4.5 Ω")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    for k, v in st.session_state.oracle.phi.items():
        st.progress(v, text=f"{k}: {v:.2f}")
    
    if st.button("🌙 Sommeil Profond"):
        st.session_state.oracle.save_all()
        github_sync()
        st.success("Mémoire consolidée.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Nourrir l'Oracle (PDF, TXT, DOCX, JSON)", type=["pdf", "txt", "docx", "json"])
if uploaded_file:
    content = ""
    if uploaded_file.type == "application/json":
        data = json.load(uploaded_file)
        content = json.dumps(data)
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        content = " ".join([p.extract_text() for p in reader.pages])
    else:
        content = uploaded_file.read().decode()
    
    if st.button("Injecter Données"):
        st.session_state.oracle.add_to_memory(content)
        st.success("Données intégrées à la base neuronale.")

# --- CHAT INTERFACE ---
for message in st.session_state.chat:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Votre question..."):
    # Affichage utilisateur
    st.session_state.chat.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Génération Oracle
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("L'Oracle analyse les probabilités..."):
            response = st.session_state.oracle.generate_response(prompt)
            # Zone de texte pour copie facile
            st.text_area("Résultat (Copie facile) :", value=response, height=200)
            st.markdown(response)
    
    st.session_state.chat.append({"role": "assistant", "content": response, "avatar": "🧠"})
    
    # Auto-save
    st.session_state.oracle.save_all()
    github_sync()
