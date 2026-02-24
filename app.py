import streamlit as st
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION GEMINI-LIKE ---
st.set_page_config(page_title="VTM Intelligence", page_icon="⚛️", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #131314; color: #e3e3e3; }
    .stChatInputContainer { background-color: #1e1f20; border-radius: 28px; border: 1px solid #444746; }
    .stChatMessage { background-color: transparent !important; }
    /* Masquer les éléments techniques */
    .stStatus { border: none !important; background: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR DE RAISONNEMENT INVISIBLE ---
class VTMBrain:
    def __init__(self, matrix_text):
        self.matrix = matrix_text

    def internal_reasoning(self, query):
        """ Calcule la stabilité en arrière-plan (sans affichage) """
        phi_c = len(query) / 10.0
        # Système dynamique TTU-MC3
        def flow(t, y):
            return [-0.6*y[0] + 1.2*y[1], -0.7*y[1] + 0.8*y[0]*y[2], 0.5*y[1]**2 - 0.3*y[2]]
        # On résout pour valider la cohérence de la pensée
        sol = solve_ivp(flow, [0, 5], [1.0, phi_c / 9.0, 0.1])
        return sol.y[0, -1] > 0.5  # Retourne si la pensée est stabilisée

    def generate_response(self, query):
        """ Synthétise une réponse claire basée sur le savoir local ou global """
        if self.matrix:
            # Recherche de résonance dans tes thèses
            segments = [s for s in self.matrix.split('.') if any(w in s.lower() for w in query.lower().split())]
            if segments:
                return f"{segments[0].strip()}. Cela s'inscrit dans la dynamique de stabilité structurelle de vos travaux."
        
        # Réponse autonome si la matrice est vide ou ne contient pas la réponse
        return "L'intelligence, dans ce contexte, est la capacité à transformer le flux d'informations du monde en une structure cohérente et stable. C'est un équilibre permanent entre la mémoire acquise et la dissipation nécessaire au renouveau."

# --- INTERFACE DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vault" not in st.session_state:
    st.session_state.vault = ""

# Sidebar discrète pour charger la connaissance
with st.sidebar:
    st.title("📂 Matrice")
    uploaded = st.file_uploader("Fichiers doctoraux", accept_multiple_files=True)
    if uploaded:
        text = ""
        for f in uploaded:
            if f.name.endswith('.pdf'):
                pdf = PdfReader(f); text += " ".join([p.extract_text() for p in pdf.pages])
            elif f.name.endswith('.docx'):
                doc = Document(f); text += " ".join([p.text for p in doc.paragraphs])
        st.session_state.vault = text
        st.success("Connaissance intégrée.")

st.title("⚛️ VTM Intelligence")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # L'IA "réfléchit" (Calcul VTM invisible)
        with st.status("Analyse en cours...", expanded=False) as status:
            brain = VTMBrain(st.session_state.vault)
            is_stable = brain.internal_reasoning(prompt)
            time.sleep(0.8)
            status.update(label="Réflexion terminée", state="complete")

        # Résultat de la réflexion
        answer = brain.generate_response(prompt)

        # Animation d'écriture fluide (Style Gemini)
        full_text = ""
        for chunk in answer.split():
            full_text += chunk + " "
            response_placeholder.markdown(full_text + "▌")
            time.sleep(0.04)
        response_placeholder.markdown(full_text)
        
    st.session_state.messages.append({"role": "assistant", "content": full_text})
