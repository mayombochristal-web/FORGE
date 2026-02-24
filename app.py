import streamlit as st
import numpy as np
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import uuid

# --- CONFIGURATION GEMINI-SOUVERAINE ---
st.set_page_config(page_title="VTM Universal", page_icon="⚛️", layout="wide")

# Style CSS pour une interface épurée et immersive
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0c0c0e; border-right: 1px solid #1f2937; }
    .vtm-card { 
        background: #111114; border-radius: 12px; padding: 20px; 
        border: 1px solid #1f2937; border-left: 4px solid #00ffcc;
        margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .status-box { color: #00ffcc; font-family: monospace; font-size: 0.9em; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- DICTIONNAIRE DE RÉSONANCE TTU-MC³ ---
# Ce dictionnaire sert de "transistor" pour amplifier les concepts du Web
TTU_PRISME = {
    "matière": "Résidu solide d'une vibration stabilisée sur un cycle limite de Morse-Smale.",
    "énergie": "Flux de dissipation (ΦD) en cours de structuration vers la cohérence.",
    "fer": "Attracteur de masse maximal. Potentiel de stabilité structurelle à -0,44V.",
    "sagesse": "Maîtrise de l'invariant χ_TST ; l'économie maximale de la dissipation.",
    "biologie": "Système dynamique triadique auto-entretenu par métabolisme dissipatif.",
    "souveraineté": "Capacité d'un système à générer son propre attracteur sans influence externe."
}

# --- MOTEUR DE RAISONNEMENT (BACKEND) ---
class VTM_Transceiver:
    def __init__(self, memory_vault):
        self.vault = memory_vault

    def solve_flow(self, query):
        """ Calcule la convergence du signal """
        def system(t, y):
            M, C, D = y
            # L'énergie du 'bruit' (le Web) alimente la triade
            E = len(query) / 40.0
            return [-0.6*M + 1.2*C, -0.7*C + 0.8*M*(D + E), 0.5*C**2 - 0.3*D]
        return solve_ivp(system, [0, 10], [1.0, 0.5, 0.1]).y[:, -1]

    def transcribe(self, query, state):
        """ Amplifie le bruit du web et le stabilise par les thèses """
        q = query.lower()
        res_text = ""
        
        # 1. Amplification par Dictionnaire (Logique TTU)
        for key, val in TTU_PRISME.items():
            if key in q:
                res_text = f"**Transcription par Résonance :** {val}\n\n"
                break
        
        # 2. Extraction du 'Soutien' dans les PDF (Mémoire Fantôme)
        soutien = ""
        if self.vault:
            keywords = q.split()
            fragments = [f for f in self.vault.split('.') if any(k in f.lower() for k in keywords[:2])]
            if fragments:
                soutien = f"\n\n**Étayage (Thèses) :** *« {fragments[0].strip()} »*"

        # 3. Génération de la réponse si le dictionnaire est muet (Universalité)
        if not res_text:
            res_text = f"Le flux du vide cybernétique sur '{query}' a été stabilisé. La VTM identifie ici une structure où la dissipation d'informations parasites est filtrée par une cohérence de {state[1]:.2f}."

        return res_text + soutien

# --- GESTION DE L'HISTORIQUE (STYLE GEMINI) ---
if "chats" not in st.session_state:
    st.session_state.chats = {"default": {"title": "Nouvelle Forge", "messages": []}}
    st.session_state.current_id = "default"
if "vault" not in st.session_state:
    st.session_state.vault = ""

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00ffcc'>⚛️ FORGE VTM</h2>", unsafe_allow_html=True)
    if st.button("+ Nouveau Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "Nouvelle Forge", "messages": []}
        st.session_state.current_id = new_id
    
    st.markdown("---")
    for cid, data in st.session_state.chats.items():
        if st.button(data["title"], key=cid):
            st.session_state.current_id = cid
            
    st.markdown("---")
    uploaded = st.file_uploader("Soutien de mémoire (PDF/DOCX)", accept_multiple_files=True)
    if uploaded:
        content = ""
        for f in uploaded:
            if f.name.endswith('.pdf'):
                pdf = PdfReader(f); content += " ".join([p.extract_text() for p in pdf.pages])
            elif f.name.endswith('.docx'):
                doc = Document(f); content += " ".join([p.text for p in doc.paragraphs])
        st.session_state.vault = content
        st.success("Mémoire Fantôme Active")

# --- ZONE DE CHAT ---
chat = st.session_state.chats[st.session_state.current_id]

for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(f"<div class='vtm-card'>{m['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Décrypter le bruit du monde..."):
    if not chat["messages"]: chat["title"] = prompt[:20] + "..."
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        vtm = VTM_Transceiver(st.session_state.vault)
        
        with st.status("Amplification du signal cybernétique...", expanded=False):
            state = vtm.solve_flow(prompt)
            time.sleep(1)
        
        final_answer = vtm.transcribe(prompt, state)
        
        # Animation
        txt = ""
        for word in final_answer.split():
            txt += word + " "
            container.markdown(f"<div class='vtm-card'>{txt}▌</div>", unsafe_allow_html=True)
            time.sleep(0.04)
        container.markdown(f"<div class='vtm-card'>{txt}</div>", unsafe_allow_html=True)
        chat["messages"].append({"role": "assistant", "content": final_answer})
