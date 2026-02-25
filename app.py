import streamlit as st
import torch
import yaml
import os
from pathlib import Path
from ttu_model import TTULanguageModel
from utils import plot_trajectory_3d
import numpy as np

# Configuration de la page
st.set_page_config(page_title="TTU-MC³ AI Chatbot", layout="wide")
st.title("🧠💬 TTU-MC³ AI - Chatbot à raisonnement augmenté")
st.markdown("Un assistant conversationnel basé sur GPT-2 + dynamique triadique dissipative.")

# Chargement config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialisation du modèle (en cache)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTULanguageModel(
        base_model_name=config['model']['name'],
        hidden_dim=config['model']['hidden_dim'],
        dt=config['model']['dt']
    ).to(device)
    return model, device

model, device = load_model()

# Session state pour la conversation
if "history" not in st.session_state:
    st.session_state.history = []  # liste de messages
    st.session_state.ttu_state = None
    st.session_state.traj = []

# Barre latérale de paramètres
st.sidebar.header("⚙️ Paramètres de génération")
temperature = st.sidebar.slider("Température", 0.1, 2.0, config['model']['temperature'], 0.1)
max_new_tokens = st.sidebar.slider("Max nouveaux tokens", 50, 500, 150, 10)
mode = st.sidebar.selectbox("Mode", ["Standard", "Dissipation active", "Silence dissipatif", "Exploration"])
if mode == "Dissipation active":
    st.sidebar.info("Mode haute créativité")
elif mode == "Silence dissipatif":
    st.sidebar.info("Mode stable, moins de bruit")
elif mode == "Exploration":
    st.sidebar.info("Mode exploratoire, plus de risque")

# Bouton pour reset conversation
if st.sidebar.button("🗑️ Nouvelle conversation"):
    st.session_state.history = []
    st.session_state.ttu_state = None
    st.session_state.traj = []
    st.rerun()

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")

    # Afficher l'historique
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input utilisateur
    prompt = st.chat_input("Posez votre question...")
    if prompt:
        # Afficher message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})

        # Génération
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                response, new_state, traj = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    ttu_state=st.session_state.ttu_state
                )
                st.markdown(response)
                st.session_state.history.append({"role": "assistant", "content": response})
                st.session_state.ttu_state = new_state
                st.session_state.traj = traj

with col2:
    st.header("📈 Visualisation TTU")
    if st.session_state.traj:
        fig = plot_trajectory_3d(st.session_state.traj)
        st.plotly_chart(fig, use_container_width=True)
        # Afficher les valeurs courantes
        last = st.session_state.traj[-1][0]
        st.metric("Cohérence (ϕ_C)", f"{last[0]:.3f}")
        st.metric("Dissipation (ϕ_D)", f"{last[1]:.3f}")
        st.metric("Mémoire (ϕ_M)", f"{last[2]:.3f}")
    else:
        st.info("Posez une question pour voir la trajectoire.")

    # Ajout d'un exemple de question mathématique
    st.subheader("Exemples")
    if st.button("Hypothèse de Riemann"):
        # On pré-remplit le champ, mais il faudrait une interaction
        st.info("Pour l'instant, je ne peux pas résoudre l'hypothèse de Riemann, mais je peux discuter de son importance.")
        # On pourrait insérer automatiquement la question
    if st.button("Théorème de Fermat"):
        st.info("Le dernier théorème de Fermat a été démontré par Andrew Wiles en 1994.")