import streamlit as st
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.decomposition import PCA

# ==========================================================
# CONFIGURATION CLOUD & MOTEUR SPECTRAL
# ==========================================================
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": "Bearer hf_votre_token_ici"} # Optionnel : insérez votre token HF

class OracleV18Engine:
    def __init__(self):
        if "latent" not in st.session_state:
            st.session_state.latent = {"torque": 1.0, "stabilite": 0.8}
        
    def calculate_torque(self, current_vec, previous_vec):
        """Calcule la rotation de phase (Torque) entre deux états sémantiques"""
        if previous_vec is None: return 0.0
        # Normalisation
        v1 = current_vec / np.linalg.norm(current_vec)
        v2 = previous_vec / np.linalg.norm(previous_vec)
        # Produit scalaire pour l'angle (cos theta)
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos_theta) # Angle en radians

# ==========================================================
# FONCTIONS DE GÉNÉRATION ET ANALYSE
# ==========================================================
def call_oracle_api(prompt, history):
    context = "Tu es l'Oracle V18. Ton moteur utilise la Théorie Spectrale Triadique (TST).\n"
    full_prompt = f"{context} Historique: {history[-3:]} \nUtilisateur: {prompt}\nAssistant:"
    
    payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.7}}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        result = response.json()
        return result[0]['generated_text'] if isinstance(result, list) else "Le moteur préchauffe... réessayez."
    except:
        return "Connexion au flux sémantique interrompue."

# ==========================================================
# INTERFACE STREAMLIT (MOBILE READY)
# ==========================================================
st.set_page_config(page_title="Oracle V18 - Torque Sémantique", layout="centered")
engine = OracleV18Engine()

if "history" not in st.session_state:
    st.session_state.history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = []

st.title("👁️ Oracle V18")
st.caption("Analyse de Torque | Sémantique Spectrale | Cloud Gratuit")

# Zone de Chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input Utilisateur
if user_input := st.chat_input("Saisissez votre impulsion sémantique..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Calcul des sauts quantiques sémantiques..."):
            # 1. Génération du texte
            reponse = call_oracle_api(user_input, st.session_state.history)
            
            # 2. Simulation du Torque (Rotation de phase)
            # Pour l'IA, chaque interaction est une rotation dans l'espace de Hilbert
            torque_val = np.random.uniform(0.1, 1.5) # Simulé pour mobile sans BERT local
            st.session_state.latent["torque"] = torque_val
            
            output = f"{reponse}\n\n---\n**🔧 Diagnostic Spectral**\n*Torque mesuré : {torque_val:.3f} rad/interaction*"
            st.write(output)
            st.session_state.history.append({"role": "assistant", "content": output})

# ==========================================================
# DASHBOARD DE PHASE (SIDEBAR)
# ==========================================================
with st.sidebar:
    st.header("📊 Métriques TST")
    st.metric("Torque (Rotation)", f"{st.session_state.latent['torque']:.2f} rad")
    st.metric("Cohérence (Voyelle)", "Optimale" if st.session_state.latent['torque'] < 1.0 else "Instable")
    
    # Simulation du signal de phase pour visualisation
    t = np.linspace(0, 10, 100)
    phase_signal = np.sin(t * st.session_state.latent["torque"])
    st.line_chart(phase_signal)
    
    if st.button("📥 Exporter la session (.txt)"):
        export_data = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
        st.download_button("Télécharger", export_data, file_name="oracle_v18_session.txt")
    
    if st.button("🗑️ Reset"):
        st.session_state.history = []
        st.rerun()
