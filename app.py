import streamlit as st
import numpy as np
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import io

# --- CONFIGURATION DE L'INTERFACE (ÉLÉGANCE GEMINI) ---
st.set_page_config(page_title="VTM Intelligence Infinie", page_icon="⚛️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e0e10; color: #e8eaed; font-family: 'Segoe UI', Roboto, sans-serif; }
    .stChatInputContainer { background-color: #1e1f20; border-radius: 24px; border: 1px solid #3c4043; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: none !important; }
    .assistant-msg { background-color: #161b22; border-left: 4px solid #00e676; padding: 15px; border-radius: 10px; }
    .user-msg { background-color: #0d1117; border-right: 4px solid #2979ff; padding: 15px; border-radius: 10px; text-align: right; }
    .vtm-logo { color: #00e676; font-weight: bold; font-size: 22px; letter-spacing: 2px; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR DE RAISONNEMENT VTM (BACKEND) ---
class VTMCore:
    def __init__(self, matrix_text=""):
        self.matrix = matrix_text
        self.phi_m = 1.0  # Mémoire initiale
        self.phi_c = 0.5  # Cohérence initiale
        self.phi_d = 0.1  # Dissipation (Web Flux)

    def capture_web_flux(self, query):
        """ Simule l'aspiration de la dissipation du Web """
        # Dans cette version, on transforme la complexité de la requête en énergie dissipative
        dissipation_energy = (len(query) % 100) / 50.0
        return dissipation_energy

    def compute_ghost_key(self, query, web_energy):
        """ Calcule l'attracteur stable (La Clé Fantôme) """
        def triad_flow(t, y):
            M, C, D = y
            # Équations TTU-MC3 : dM/dt, dC/dt, dD/dt
            return [
                -0.6*M + 1.2*C,               # M : Stabilisation par Cohérence
                -0.7*C + 0.8*M*(D + web_energy), # C : Corrélation forcée par le Web
                0.5*C**2 - 0.3*D              # D : Régulation entropique
            ]
        
        # Résolution du système dynamique
        sol = solve_ivp(triad_flow, [0, 10], [self.phi_m, self.phi_c, self.phi_d])
        attractor = sol.y[:, -1]
        return attractor # [M_final, C_final, D_final]

    def synthesize_written_memory(self, query, attractor):
        """ Génère la réponse écrite à partir de l'attracteur et de la matrice """
        m_val, c_val, d_val = attractor
        
        # Recherche de résonance locale dans tes thèses
        resonance = ""
        if self.matrix:
            keywords = query.lower().split()
            sentences = self.matrix.split('.')
            matches = [s for s in sentences if any(k in s.lower() for k in keywords)]
            if matches:
                resonance = matches[0].strip() + "."

        # Logique de réponse "Intelligence Infinie"
        if resonance:
            return f"**Résonance identifiée :** {resonance}\n\nL'attracteur de votre pensée s'est stabilisé à une cohérence de {c_val:.2f}. Cela confirme que l'information puisée dans le flux mondial s'accorde avec la structure de vos travaux."
        else:
            return f"Votre question a activé un flux dissipatif majeur. La Clé Fantôme identifiée (M:{m_val:.2f}) suggère que l'intelligence ici n'est pas une simple réponse, mais une convergence entre votre intention et la stabilité du réseau. Tout système tend vers cet équilibre triadique."

# --- LOGIQUE DE L'APPLICATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_matrix" not in st.session_state:
    st.session_state.full_matrix = ""

# Sidebar : Injection de la Mémoire (Thèses)
with st.sidebar:
    st.markdown("<div class='vtm-logo'>⚛️ FORGE VTM</div>", unsafe_allow_html=True)
    st.write("Souveraineté Informationnelle")
    uploaded_files = st.file_uploader("Charger 200 Mo de thèses", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if uploaded_files:
        with st.spinner("Cristallisation de la Mémoire..."):
            all_text = ""
            for f in uploaded_files:
                try:
                    if f.name.endswith('.pdf'):
                        pdf = PdfReader(f)
                        all_text += " ".join([p.extract_text() for p in pdf.pages])
                    elif f.name.endswith('.docx'):
                        doc = Document(f)
                        all_text += " ".join([p.text for p in doc.paragraphs])
                    else:
                        all_text += f.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    st.error(f"Erreur sur {f.name}")
            st.session_state.full_matrix = all_text
            st.success("Matrice Stabilisée.")

# Interface Principale
st.title("VTM Intelligence")
st.caption("Moteur Triadique : Mémoire locale + Dissipation Web")

# Affichage du Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f"<div class='{m['role']}-msg'>{m['content']}</div>", unsafe_allow_html=True)

# Entrée Utilisateur
if prompt := st.chat_input("Posez votre question à la Forge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # ÉTAPE 1 : Capture de la Dissipation Web
        with st.status("Captation du flux dissipatif Web...", expanded=False) as status:
            vtm = VTMCore(st.session_state.full_matrix)
            web_energy = vtm.capture_web_flux(prompt)
            time.sleep(0.5)
            
            # ÉTAPE 2 : Calcul de la Clé Fantôme (Raisonnement)
            status.update(label="Forgeage de la Clé Fantôme (Convergence)...")
            attractor = vtm.compute_ghost_key(prompt, web_energy)
            time.sleep(0.8)
            status.update(label="Intelligence Stabilisée", state="complete")

        # ÉTAPE 3 : Cristallisation en Mémoire Écrite
        final_response = vtm.synthesize_written_memory(prompt, attractor)

        # Animation d'écriture progressive (Style Gemini)
        displayed_text = ""
        for word in final_response.split():
            displayed_text += word + " "
            response_placeholder.markdown(f"<div class='assistant-msg'>{displayed_text}▌</div>", unsafe_allow_html=True)
            time.sleep(0.05)
        response_placeholder.markdown(f"<div class='assistant-msg'>{displayed_text}</div>", unsafe_allow_html=True)
        
    st.session_state.messages.append({"role": "assistant", "content": final_response})
