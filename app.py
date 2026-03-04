import streamlit as st
import base64
import requests
import PyPDF2
import docx
import json
from oracle_core import OracleBrain

# --- CONFIGURATION ---
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
MEM_FILE = "oracle_memory.json"

def github_sync():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        with open(MEM_FILE, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=5)
        sha = r.json()["sha"] if r.status_code == 200 else None
        data = {"message": "🧬 Sync Oracle", "content": content, "branch": "main"}
        if sha:
            data["sha"] = sha
        requests.put(url, headers=headers, json=data, timeout=10)
    except Exception as e:
        st.sidebar.error(f"Erreur GitHub : {e}")

@st.cache_resource
def init_oracle():
    return OracleBrain(MEM_FILE)

if "oracle" not in st.session_state:
    st.session_state.oracle = init_oracle()
    st.session_state.chat = []
    st.session_state.strict_mode = False

st.set_page_config(page_title="ORACLE V5.2 Ω", page_icon="🧠", layout="centered")
st.title("🧠 ORACLE V5.2 Ω — Assistant documentaire")

with st.sidebar:
    st.header("⚙️ Paramètres")
    k = st.slider("Nombre de passages analysés", min_value=1, max_value=10, value=4)
    st.session_state.strict_mode = st.checkbox("Mode strict (réponse uniquement si documentée)", value=st.session_state.strict_mode)
    st.divider()
    st.header("🧬 État neuronal")
    for k_, v in st.session_state.oracle.phi.items():
        st.caption(f"{k_.upper()}: {v:.2f}")
        st.progress(v)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Sauvegarder"):
            try:
                st.session_state.oracle.save_all()
                github_sync()
                st.success("Mémoire sauvegardée et synchronisée.")
            except Exception as e:
                st.error(f"Erreur sauvegarde : {e}")
    with col2:
        if st.button("🗑️ Effacer chat"):
            st.session_state.chat = []
            st.rerun()
    if st.session_state.chat:
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat])
        st.download_button("📥 Télécharger la conversation", data=chat_text,
                           file_name="oracle_conversation.txt", mime="text/plain")

with st.expander("📥 Injecter un document dans la mémoire"):
    uploaded_file = st.file_uploader("Choisir un fichier (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    if uploaded_file is not None and st.button("📄 Assimiler le document"):
        with st.spinner("Analyse et découpage en cours..."):
            try:
                if uploaded_file.type == "application/pdf":
                    reader = PyPDF2.PdfReader(uploaded_file)
                    text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = docx.Document(uploaded_file)
                    text = "\n".join([p.text for p in doc.paragraphs])
                else:
                    text = uploaded_file.read().decode("utf-8", errors="ignore")
                added = st.session_state.oracle.add_to_memory(text)
                st.success(f"✅ Document assimilé ! {added} nouveaux passages uniques.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"], avatar=msg.get("avatar", "🧠")):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources consultées"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"**Extrait {i+1}:** {src[:300]}..." if len(src) > 300 else src)

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.chat.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧠"):
        with st.status("🔍 Recherche dans les documents...", expanded=True) as status:
            context_chunks = st.session_state.oracle.search_memory(prompt, k=k)
            if not context_chunks:
                st.info("Aucun passage pertinent trouvé dans les documents.")
            status.update(label="✍️ Génération de la réponse...")
            try:
                response, sources = st.session_state.oracle.generate_response(
                    prompt,
                    context_chunks=context_chunks,
                    strict_mode=st.session_state.strict_mode
                )
                st.markdown(response)
                if sources:
                    with st.expander("📚 Sources consultées"):
                        for i, src in enumerate(sources):
                            st.caption(f"**Extrait {i+1}:** {src[:300]}..." if len(src) > 300 else src)
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")
                response = "Désolé, une erreur technique est survenue."
                sources = []
    st.session_state.chat.append({
        "role": "assistant",
        "content": response,
        "avatar": "🧠",
        "sources": context_chunks if 'context_chunks' in locals() else []
    })
    try:
        st.session_state.oracle.save_all()
        github_sync()
    except:
        pass