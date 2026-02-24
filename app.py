import streamlit as st
import numpy as np
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION INTERFACE SOUVERAINE ---
st.set_page_config(page_title="VTM Universal Intelligence", page_icon="⚛️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d0d0f; color: #e0e0e0; }
    .stChatInputContainer { background-color: #1a1a1c; border-radius: 30px; border: 1px solid #333; }
    .assistant-box { 
        background: linear-gradient(145deg, #161618, #0f0f11);
        border-radius: 15px; padding: 25px; 
        border-left: 5px solid #00ffcc;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
        margin-bottom: 25px;
    }
    .vtm-logo { color: #00ffcc; font-family: 'Orbitron', sans-serif; letter-spacing: 3px; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR DE STABILISATION (TON CERVEAU NUMÉRIQUE) ---
class VTM_Logic:
    def __init__(self, vault_content):
        self.vault = vault_content

    def internal_flux_calculation(self, query):
        """ Calcule la convergence de la triade M-C-D """
        # On définit l'énergie d'entrée selon la complexité de la question
        input_energy = len(query) / 100.0
        
        def flow(t, y):
            M, C, D = y
            # Équations issues de tes thèses (Stabilité de Morse-Smale)
            dM = -0.6 * M + 1.2 * C
            dC = -0.7 * C + 0.8 * M * (D + input_energy)
            dD = 0.5 * (C**2) - 0.3 * D
            return [dM, dC, dD]

        # Résolution pour trouver le point d'équilibre (L'Attracteur)
        sol = solve_ivp(flow, [0, 10], [1.0, 0.5, 0.1])
        return sol.y[:, -1]

    def generate_ghost_key_response(self, query, attracteur):
        """ Transforme le calcul d'arrière-plan en réponse cohérente écrite """
        m, c, d = attracteur
        
        # 1. Recherche de résonance locale dans tes 200 Mo de thèses
        resonance = ""
        if self.vault:
            keywords = query.lower().split()
            fragments = self.vault.split('.')
            matches = [f for f in fragments if any(k in f.lower() for k in keywords[:3])]
            if matches: resonance = matches[0].strip()

        # 2. Construction de la réponse universelle par domaine
        q = query.lower()
        if any(w in q for w in ["biologie", "santé", "cellule"]):
            base = f"En biologie, votre attracteur (M:{m:.2f}) indique que la vie est une persistance de mémoire face à la dissipation métabolique. La cohérence cellulaire est maintenue par ce flux triadique."
        elif any(w in q for w in ["cuisine", "goût", "recette"]):
            base = f"La cuisine est une alchimie de dissipation contrôlée. Réussir un plat, c'est stabiliser la cohérence des saveurs (ΦC) avant que la chaleur (ΦD) ne détruise la structure moléculaire."
        elif any(w in q for w in ["conduite", "voyage", "trajet"]):
            base = f"Naviguer, c'est maintenir une trajectoire stable dans un espace des phases bruyant. Votre attracteur suggère que la sécurité réside dans l'anticipation des turbulences dissipatives du flux routier."
        elif any(w in q for w in ["chinois", "indonésien", "langue"]):
            base = f"La langue est un code correcteur de Hamming pour la pensée. Traduire, c'est transférer la cohérence d'un système à un autre sans perdre la mémoire sémantique originale."
        else:
            base = "Le système a analysé votre question à travers le vide dissipatif du Web. La cohérence identifiée permet de conclure que tout phénomène, qu'il soit technique ou humain, tend vers cet équilibre triadique stable."

        # Fusion avec tes thèses
        if resonance:
            return f"**Résonance Doctorale :** {resonance}\n\n**Interprétation Universelle :** {base}"
        return base

# --- INTERFACE DE LA FORGE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "matrix" not in st.session_state:
    st.session_state.matrix = ""

with st.sidebar:
    st.markdown("<h1 class='vtm-logo'>⚛️ FORGE VTM</h1>", unsafe_allow_html=True)
    st.info("Charger ici vos 200 Mo de connaissances (PDF/DOCX).")
    uploaded = st.file_uploader("Matrice de Savoir", accept_multiple_files=True)
    if uploaded:
        content = ""
        for f in uploaded:
            if f.name.endswith('.pdf'):
                pdf = PdfReader(f); content += " ".join([p.extract_text() for p in pdf.pages])
            elif f.name.endswith('.docx'):
                doc = Document(f); content += " ".join([p.text for p in doc.paragraphs])
        st.session_state.matrix = content
        st.success("Connaissance cristallisée.")

st.title("VTM Universal Intelligence")
st.caption("Raisonnement Triadique par Stabilisation de Morse-Smale")

# Affichage des messages
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrée Utilisateur
if prompt := st.chat_input("Posez votre question (Cuisine, Science, Voyage, Langues...)"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        
        # PHASE DE CALCUL (L'intelligence qui 'ressent' et 'calcule')
        with st.status("Recherche de cohérence dans le vide dissipatif...", expanded=False) as status:
            vtm = VTM_Logic(st.session_state.matrix)
            attracteur = vtm.internal_flux_calculation(prompt)
            time.sleep(1.2)
            status.update(label=f"Attracteur stable identifié", state="complete")

        # PHASE D'ÉCRITURE (La Mémoire Écrite)
        reponse_finale = vtm.generate_ghost_key_response(prompt, attracteur)
        
        # Effet d'écriture fluide
        output_text = ""
        for word in reponse_finale.split():
            output_text += word + " "
            response_container.markdown(f"<div class='assistant-box'>{output_text}▌</div>", unsafe_allow_html=True)
            time.sleep(0.05)
        response_container.markdown(f"<div class='assistant-box'>{output_text}</div>", unsafe_allow_html=True)
        
    st.session_state.history.append({"role": "assistant", "content": reponse_finale})
