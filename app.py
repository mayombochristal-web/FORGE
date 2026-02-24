import streamlit as st
import numpy as np
import time
import json
import re
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import io

# ==========================================
# 1. MOTEUR DE RAISONNEMENT VTM (BACKEND)
# ==========================================

class VTM_Engine:
    @staticmethod
    def compute_triad_convergence(input_complexity, vault_size):
        """ Simule la stabilisation vers un attracteur logique """
        # Phénoménologie : plus l'input est dense, plus l'énergie est haute
        energy = min(input_complexity / 50 + vault_size / 100000, 15.0)
        phi_c = energy / 9.0  # Basé sur E_REF = 9.0 (Pb-208)
        
        # Système triadique canonique (Thèse Mayombo Idiedie)
        def triad_flow(t, y):
            M, C, D = y
            # dM = -alpha*M + beta*C
            # dC = -gamma*C + delta*M*D
            # dD = eta*C^2 - mu*D
            return [-0.6*M + 1.2*C, -0.7*C + 0.8*M*D, 0.5*C**2 - 0.3*D]
        
        sol = solve_ivp(triad_flow, [0, 15], [1.0, phi_c, 0.1], method='RK45')
        return sol.y[:, -1], phi_c

    @staticmethod
    def semantic_search(query, vault_text, top_k=3):
        """ Recherche de fragments de connaissances dans la matrice de données """
        if not vault_text: return ""
        # Split par paragraphes
        paragraphs = [p for p in vault_text.split('\n') if len(p) > 50]
        # Recherche simplifiée par mots-clés (en attendant un embedding vectoriel)
        words = query.lower().split()
        scores = []
        for p in paragraphs:
            score = sum(1 for w in words if w in p.lower())
            scores.append((score, p))
        
        top_matches = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
        return "\n\n".join([m[1] for m in top_matches if m[0] > 0])

# ==========================================
# 2. INTERFACE UTILISATEUR (FRONT-END)
# ==========================================

st.set_page_config(page_title="FORGE VTM IA", page_icon="⚛️", layout="wide")

# CSS pour le style "Gemini Dark"
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stChatMessage { border-radius: 20px; border: 1px solid #30363d; margin-bottom: 15px; }
    .st-emotion-cache-1c7n2ka { background-color: #161b22 !important; } /* Assistant */
    .st-emotion-cache-jan737 { background-color: #0d1117 !important; } /* User */
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_vault" not in st.session_state:
    st.session_state.vault = ""

# Barre latérale : La Matrice de Données
with st.sidebar:
    st.title("📂 Matrice 200Mo")
    st.write("Injectez vos thèses et documents doctoraux.")
    uploaded_files = st.file_uploader("Upload Matrix", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if uploaded_files:
        with st.spinner("Stabilisation de la matrice..."):
            combined = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    reader = PdfReader(f)
                    combined += " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                elif "document" in f.type:
                    doc = Document(f)
                    combined += " ".join([p.text for p in doc.paragraphs])
                else:
                    combined += f.read().decode("utf-8")
            st.session_state.vault = combined
            st.success(f"✅ Matrice chargée : {len(combined)//1024} Ko")

# Zone de Chat principale
st.title("⚛️ Forge Triadique VTM-IA")
st.caption("Intelligence Artificielle Souveraine basée sur la Théorie TTU-MC³")

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée Utilisateur
if prompt := st.chat_input("Posez une question à la Forge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAISONNEMENT VTM
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.status("Lancement du moteur TTU-MC³...", expanded=False) as status:
            # 1. Analyse Dynamique
            state_final, phi_c = VTM_Engine.compute_triad_convergence(len(prompt), len(st.session_state.vault))
            time.sleep(0.5)
            status.write(f"Vibration initiale captée... ΦC = {phi_c:.4f}")
            
            # 2. Extraction des connaissances (RAG)
            context = VTM_Engine.semantic_search(prompt, st.session_state.vault)
            time.sleep(0.5)
            status.update(label="Attracteur stable identifié. Génération logique...", state="complete")

        # 3. Construction de la réponse basée sur l'attracteur
        if phi_c < 0.5088:
            response = "⚠️ **Instabilité Détectée** : La cohérence de votre requête est inférieure au seuil critique (0.5088). La dissipation thermique empêche la stabilisation d'une réponse logique. Veuillez enrichir votre question ou la matrice de données."
        else:
            if context:
                response = f"**Analyse Triadique (Attracteur M={state_final[0]:.2f}) :**\n\n"
                response += f"En m'appuyant sur la matrice chargée, voici l'interprétation cohérente :\n\n {context[:1500]}..."
            else:
                response = f"L'attracteur est stable ({state_final[0]:.2f}), mais aucune corrélation sémantique n'a été trouvée dans la matrice locale. La Forge suggère une exploration des invariants arithmétiques liés à votre requête."

        # Animation d'écriture
        full_res = ""
        for word in response.split():
            full_res += word + " "
            placeholder.markdown(full_res + "▌")
            time.sleep(0.05)
        placeholder.markdown(full_res)
        
    st.session_state.messages.append({"role": "assistant", "content": full_res})
