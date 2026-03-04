import streamlit as st
import base64
import requests
import PyPDF2
import docx
import json
from oracle_core import OracleBrain

# --- CONFIG ---
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
        r = requests.get(url, headers=headers, timeout=5)
        sha = r.json()["sha"] if r.status_code == 200 else None
        data = {"message": "🧬 Sync Oracle", "content": content, "branch": "main"}
        if sha: data["sha"] = sha
        requests.put(url, headers=headers, json=data, timeout=10)
    except: pass

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain(MEM_FILE)
    st.session_state.chat = []

st.set_page_config(page_title="ORACLE V4.5 Ω", page_icon="🧠", layout="centered")
st.title("🧠 ORACLE V4.5 Ω")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ État Neuronal")
    for k, v in st.session_state.oracle.phi.items():
        st.caption(f"{k.upper()}: {v:.2f}")
        st.progress(v)
    
    if st.button("🌙 Sauvegarder"):
        st.session_state.oracle.save_all()
        github_sync()
        st.success("Synchronisé.")

# --- INJECTION ---
with st.expander("📥 Injecter un Document"):
    uploaded_file = st.file_uploader("Fichier", type=["pdf", "txt", "docx"])
    if uploaded_file and st.button("Assimiler"):
        with st.spinner("Analyse..."):
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.session_state.oracle.add_to_memory(text)
            st.success("Mémoire mise à jour.")

# --- CHAT ---
for msg in st.session_state.chat:
    with st.chat_message(msg["role"], avatar=msg["avatar"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Votre question..."):
    st.session_state.chat.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        res_area = st.empty()
        with st.spinner("Recherche documentaire..."):
            response = st.session_state.oracle.generate_response(prompt)
            res_area.markdown(response)
    
    st.session_state.chat.append({"role": "assistant", "content": response, "avatar": "🧠"})
    st.session_state.oracle.save_all()
