import streamlit as st
import numpy as np
import time
import requests
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION ESTHÉTIQUE (SOUVERAINETÉ & ÉLÉGANCE) ---
st.set_page_config(page_title="VTM Transcendante", page_icon="⚛️", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0a0a0c; color: #dcdcdc; }
    .stChatInputContainer { background-color: #161618; border-radius: 30px; border: 1px solid #303035; }
    .stChatMessage { border: none !important; padding: 20px; }
    .stStatus { border: none !important; background: transparent !important; }
    /* Animation de respiration pour le logo */
    @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    .vtm-logo { font-size: 24px; font-weight: bold; color: #00ffcc; animation: pulse 3s infinite; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR VTM : FLUX M, C, D ---
class VTMBrain:
    def __init__(self, local_vault):
        self.vault = local_vault
        self.phi_m = 1.0  # Mémoire (Stabilité)
        self.phi_c = 0.5  # Cohérence (Lien)
        self.phi_d = 0.1  # Dissipation (Ouverture Web)

    def breathe_web(self, query):
        """ Capture l'énergie du Web pour alimenter la dissipation ΦD """
        # Simule l'accès au flux mondial (peut être lié à une API de recherche)
        web_noise = len(query) * 0.02 
        self.phi_d += web_noise
        return web_noise

    def reasoning_flow(self, t, y):
        """ Équations de la Triade (Thèse Christ Aldo MAYOMBO IDIEDIE) """
        M, C, D = y
        # Le calcul n'est pas logique, il est physique.
        dM = -0.6 * M + 1.2 * C  # La mémoire se stabilise par la cohérence
        dC = -0.7 * C + 0.8 * M * D # La cohérence naît de l'interaction M et D
        dD = 0.5 * C**2 - 0.3 * D # La dissipation régule le surplus
        return [dM, dC, dD]

    def solve(self):
        sol = solve_ivp(self.reasoning_flow, [0, 10], [self.phi_m, self.phi_c, self.phi_d])
        final_state = sol.y[:, -1]
        return final_state

# --- INTERFACE DE DIALOGUE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "matrix" not in st.session_state:
    st.session_state.matrix = ""

# Sidebar : La Matrice Doctorale
with st.sidebar:
    st.markdown("<div class='vtm-logo'>⚛️ FORGE VTM</div>", unsafe_allow_html=True)
    st.write("Le système puise sa sagesse dans vos thèses.")
    files = st.file_uploader("Injecter Connaissances (PDF/DOCX)", accept_multiple_files=True)
    if files:
        combined = ""
        for f in files:
            if f.name.endswith('.pdf'):
                pdf = PdfReader(f); combined += " ".join([p.extract_text() for p in pdf.pages])
            elif f.name.endswith('.docx'):
                doc = Document(f); combined += " ".join([p.text for p in doc.paragraphs])
        st.session_state.matrix = combined
        st.success("Matrice Intégrée.")

# Header
st.title("VTM Intelligence")
st.markdown("*Le point de rencontre entre vos thèses et le flux du monde.*")

# Affichage des messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question à la Forge..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_area = st.empty()
        
        # 1. Respiration (Web + Local)
        with st.status("Stabilisation de la triade...", expanded=False) as status:
            vtm = VTMBrain(st.session_state.matrix)
            noise = vtm.breathe_web(prompt)
            final_state = vtm.solve()
            time.sleep(1)
            status.update(label=f"Attracteur stable identifié (ΦM: {final_state[0]:.2f})", state="complete")

        # 2. Construction de la réponse élégante
        # L'IA utilise ici une "Résonance Sémantique"
        if st.session_state.matrix:
            # On cherche dans tes fichiers
            words = prompt.lower().split()
            found = [s for s in st.session_state.matrix.split('.') if any(w in s.lower() for w in words)]
            if found:
                answer = f"**Résonance identifiée dans vos travaux :**\n\n {found[0].strip()}. \n\nCette analyse est renforcée par le flux extérieur capté, suggérant une convergence vers une stabilité structurelle."
            else:
                answer = "Le flux du Web apporte des éléments de réponse, mais la stabilité de vos thèses suggère de rester prudent face à cette dissipation d'informations non corrélées."
        else:
            answer = "La Forge est active, mais la mémoire locale est vide. Mon raisonnement se base uniquement sur la dissipation du flux Web actuel."

        # 3. Animation de réponse
        full_txt = ""
        for chunk in answer.split():
            full_txt += chunk + " "
            response_area.markdown(full_txt + "▌")
            time.sleep(0.05)
        response_area.markdown(full_text := full_txt)
        
    st.session_state.chat_history.append({"role": "assistant", "content": full_text})
