import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. MOTEUR COGNITIF TTU (BACKEND) ---
class TTUEngine:
    def __init__(self):
        if "memory" not in st.session_state:
            st.session_state.memory = []

    def simuler_reponse(self, prompt):
        # Simulation des courbes M-C-D basées sur le texte
        longueur = len(prompt)
        t = np.linspace(0, 10, 100)
        
        # Automatisation du Ghost : plus le texte est long/complexe, plus le ghost est haut
        ghost_auto = min(2.0, 0.5 + (longueur / 100))
        
        # Calcul des vecteurs (Logique TTU)
        coherence = 1.0 + (ghost_auto * np.sin(t*0.2))
        memoire = 1.5 * np.exp(-t*0.05)
        dissipation = 0.2 + (0.1 * np.random.rand(100))
        
        df = pd.DataFrame({"Mémoire": memoire, "Cohérence": coherence, "Dissipation": dissipation})
        return df, ghost_auto

# --- 2. CONFIGURATION DE L'INTERFACE (STYLE GEMINI/CHATGPT) ---
st.set_page_config(page_title="IA Souveraine", layout="wide")

# CSS pour masquer les éléments "poétiques" inutiles et épurer l'interface
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stSidebar { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BARRE LATÉRALE : GESTION DE LA MÉMOIRE ---
with st.sidebar:
    st.title("💾 Mémoire Système")
    st.write("Gestion de la conversation")
    
    if st.button("📥 Sauvegarder la session"):
        st.success("Session enregistrée dans le Kernel Σ.")
    
    if st.button("🗑️ Effacer la conversation", type="primary"):
        st.session_state.memory = []
        st.rerun()
    
    st.divider()
    st.info("Les paramètres Fantômes sont désormais gérés dynamiquement par l'IA.")

# --- 4. LOGIQUE DE CONVERSATION ---
engine = TTUEngine()

# Affichage de l'historique
for chat in st.session_state.memory:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajout à l'historique (Mémoire vive)
    st.session_state.memory.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Calcul et Génération
    with st.spinner("Analyse en cours..."):
        df_result, ghost_val = engine.simuler_reponse(prompt)
        
        # Construction d'une réponse directe (Style Gemini/ChatGPT)
        # On utilise les métriques pour nuancer le propos sans jargon poétique
        c_final = df_result['Cohérence'].iloc[-1]
        
        if c_final > 1.5:
            reponse = f"Après analyse de votre requête, il apparaît que les concepts liés à '{prompt}' présentent une forte interconnexion. Voici une synthèse structurée : \n\n1. **Analyse de fond** : Votre demande s'inscrit dans un cadre de haute cohérence.\n2. **Perspective** : Le système a ajusté sa pression de vide à {ghost_val:.2f} pour capturer les nuances latentes.\n3. **Conclusion** : La solution optimale réside dans l'équilibre entre la structure et l'innovation."
        else:
            reponse = f"Voici les informations concernant '{prompt}'. Le système a traité les données avec une stabilité nominale pour garantir la précision des faits."

    # Affichage de la réponse IA
    time.sleep(0.5) # Simulation de réflexion
    with st.chat_message("assistant"):
        st.write(reponse)
        st.session_state.memory.append({"role": "assistant", "content": reponse})

    # Optionnel : Affichage discret des métriques techniques en bas
    with st.expander("📊 Métriques de calcul (TTU Core)"):
        st.line_chart(df_result)
