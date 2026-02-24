import streamlit as st
import numpy as np
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import pandas as pd

# --- CONFIGURATION ENGINE VTM ---
class VTM_Brain:
    def __init__(self, vault_text):
        self.vault_text = vault_text
        self.E_REF = 9.0
        self.PHI_SEUIL = 0.5088

    def analyze_stability(self, query):
        # L'énergie est corrélée à la longueur et la densité lexicale
        energy = len(query) / 15.0
        phi_c = energy / self.E_REF
        
        # Simulation du flot triadique (M, C, D)
        def flow(t, y):
            M, C, D = y
            return [-0.6*M + 1.2*C, -0.7*C + 0.8*M*D, 0.5*C**2 - 0.3*D]
        
        sol = solve_ivp(flow, [0, 10], [1.0, phi_c, 0.1])
        return sol.y[:, -1], phi_c

    def interpret_vault(self, query):
        if not self.vault_text:
            return "Matrice vide. La Forge résonne dans le vide."
        
        # RAG Local : Recherche de résonance sémantique
        keywords = query.lower().split()
        segments = self.vault_text.split('.')
        scored_segments = []
        for seg in segments:
            score = sum(1 for word in keywords if word in seg.lower())
            if score > 0:
                scored_segments.append((score, seg.strip()))
        
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        return " ".join([s[1] for s in scored_segments[:3]])

# --- INTERFACE FRONT-END (STYLE GEMINI) ---
st.set_page_config(page_title="VTM Forge IA", layout="wide", page_icon="⚛️")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #D1D1D1; }
    .chat-bubble { padding: 20px; border-radius: 15px; margin: 10px 0; border: 1px solid #333; }
    .assistant { background-color: #111; border-left: 5px solid #00ffcc; }
    .user { background-color: #1a1a1a; border-right: 5px solid #ff0055; text-align: right; }
    </style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vault_data" not in st.session_state:
    st.session_state.vault_data = ""

# Sidebar pour les 200 Mo de fichiers
with st.sidebar:
    st.title("📂 Matrice de Connaissance")
    files = st.file_uploader("Injecter Thèses/Cours (PDF, DOCX)", accept_multiple_files=True)
    if files:
        combined = ""
        for f in files:
            if f.type == "application/pdf":
                reader = PdfReader(f)
                combined += " ".join([p.extract_text() for p in reader.pages])
            elif "document" in f.type:
                doc = Document(f)
                combined += " ".join([p.text for p in doc.paragraphs])
            combined += "\n"
        st.session_state.vault_data = combined
        st.success("✅ Matrice stabilisée.")

# Zone de Chat
st.title("🧠 Forge VTM-IA : Transcendance Logique")
st.write("Posez votre question. Si elle est 'stupide', la dissipation thermique fera le reste.")

for chat in st.session_state.chat_history:
    role_class = "assistant" if chat["role"] == "assistant" else "user"
    st.markdown(f"<div class='chat-bubble {role_class}'>{chat['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Exprimez votre pensée..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Lancement du moteur VTM
    brain = VTM_Brain(st.session_state.vault_data)
    
    with st.status("Recherche de l'attracteur triadique...", expanded=False) as status:
        state, phi_c = brain.analyze_stability(prompt)
        time.sleep(0.8)
        status.update(label=f"Cohérence ΦC: {phi_c:.4f} | État stable trouvé.", state="complete")

    # Interprétation
    resonance = brain.interpret_vault(prompt)
    
    # Logique de réponse "Pragmatisme vs Non-sens"
    if phi_c < brain.PHI_SEUIL:
        response = f"**Bruit Thermique Détecté.** La cohérence ({phi_c:.2f}) est trop basse. La question s'évapore dans la dissipation $\Phi_D$. En TTU, cela signifie que votre requête n'a pas de masse informationnelle."
    else:
        if resonance:
            response = f"**Résonance Captée.** L'attracteur M={state[0]:.2f} pointe vers ces segments de vos thèses : \n\n > {resonance}"
        else:
            response = f"**Attracteur Isolé.** Le système est stable ({phi_c:.2f}), mais ne trouve aucune résonance dans vos fichiers. Vous explorez un vide quantique de la Forge."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()
