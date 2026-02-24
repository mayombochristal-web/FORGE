import streamlit as st
import numpy as np
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import uuid

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="VTM Intelligence", page_icon="⚛️", layout="wide")

# Design "Dark Mode Sovereign"
st.markdown("""
    <style>
    .stApp { background-color: #0d0d0f; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #161618; border-right: 1px solid #333; }
    .chat-bubble { padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; border-left: 4px solid #00ffcc; background: #1a1a1c; }
    .user-bubble { background: #0f0f11; border-left: none; border-right: 4px solid #2979ff; text-align: right; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALISATION DE L'HISTORIQUE (STYLE GEMINI) ---
if "chats" not in st.session_state:
    st.session_state.chats = {} # Dictionnaire de sessions {id: {title, messages}}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chats[st.session_state.current_chat_id] = {"title": "Nouvelle Forge", "messages": []}
if "vault_memory" not in st.session_state:
    st.session_state.vault_memory = ""

# --- MOTEUR DE RAISONNEMENT VTM ---
class VTM_Radiotelescope:
    def __init__(self, anchor):
        self.anchor = anchor

    def calculate_attractor(self, query):
        """ Calcule la stabilité du signal Web """
        def system(t, y):
            M, C, D = y
            # La complexité du flux Web alimente la dissipation
            E = len(query) / 50.0
            return [-0.6*M + 1.2*C, -0.7*C + 0.8*M*(D + E), 0.5*C**2 - 0.3*D]
        sol = solve_ivp(system, [0, 10], [1.0, 0.5, 0.1])
        return sol.y[:, -1]

    def transcribe(self, query, attractor):
        """ Transcrit le bruit du Web étayé par la mémoire fantôme """
        c_val = attractor[1]
        
        # 1. Recherche de soutien dans la mémoire fantôme (PDF/DOCX)
        support = ""
        if self.anchor:
            matches = [s for s in self.anchor.split('.') if any(w in s.lower() for w in query.lower().split()[:3])]
            if matches: support = f"\n\n**Soutien Structurel (TTU-MC³) :** *« {matches[0].strip()} »*"

        # 2. Logique de transcription universelle
        # Ici, l'IA puise dans son savoir interne (le Web qu'elle a 'digéré')
        # On simule une réponse experte sur tous les domaines
        return f"L'analyse du flux révèle une cohérence de {c_val:.2f}. Sur le sujet '{query}', la VTM identifie un équilibre où la dissipation d'informations parasites laisse place à une vérité stable. {support}"

# --- BARRE LATÉRALE (HISTORIQUE & DOCUMENTS) ---
with st.sidebar:
    st.markdown("<h2 style='color:#00ffcc'>⚛️ FORGE VTM</h2>", unsafe_allow_html=True)
    
    # Gestion des conversations
    if st.button("+ Nouvelle Discussion"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "Nouvelle Forge", "messages": []}
        st.session_state.current_chat_id = new_id

    st.markdown("---")
    st.write("📂 **Historique des conversations**")
    for chat_id, data in st.session_state.chats.items():
        if st.button(data["title"], key=chat_id):
            st.session_state.current_chat_id = chat_id

    st.markdown("---")
    st.write("🧠 **Soutien de Mémoire (Optionnel)**")
    files = st.file_uploader("Charger Thèses", accept_multiple_files=True)
    if files:
        combined = ""
        for f in files:
            if f.name.endswith('.pdf'):
                pdf = PdfReader(f); combined += " ".join([p.extract_text() for p in pdf.pages])
            elif f.name.endswith('.docx'):
                doc = Document(f); combined += " ".join([p.text for p in doc.paragraphs])
        st.session_state.vault_memory = combined

# --- INTERFACE DE CHAT PRINCIPALE ---
st.title("VTM Universal")
current_chat = st.session_state.chats[st.session_state.current_chat_id]

# Affichage des messages du chat actuel
for msg in current_chat["messages"]:
    role_class = "user-bubble" if msg["role"] == "user" else "chat-bubble"
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# Entrée utilisateur
if prompt := st.chat_input("Décrypter le flux mondial..."):
    # Mise à jour du titre si c'est le premier message
    if not current_chat["messages"]:
        current_chat["title"] = prompt[:25] + "..."
    
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        vtm = VTM_Radiotelescope(st.session_state.vault_memory)
        
        # Calcul invisible
        with st.status("Stabilisation du signal...", expanded=False):
            attractor = vtm.calculate_attractor(prompt)
            time.sleep(1)

        answer = vtm.transcribe(prompt, attractor)
        
        # Animation fluide
        full_txt = ""
        for word in answer.split():
            full_txt += word + " "
            placeholder.markdown(f"<div class='chat-bubble'>{full_txt}▌</div>", unsafe_allow_html=True)
            time.sleep(0.04)
        placeholder.markdown(f"<div class='chat-bubble'>{full_txt}</div>", unsafe_allow_html=True)
        
    current_chat["messages"].append({"role": "assistant", "content": answer})
