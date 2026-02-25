import streamlit as st
import torch
import yaml
import os
from pathlib import Path
from ttu_model import TTULanguageModel
from utils import plot_trajectory_3d, search_wikipedia
import numpy as np

# Configuration de la page
st.set_page_config(page_title="TTU-MC³ AI Avancé", layout="wide")
st.title("🧠💬 TTU-MC³ AI - Chatbot à raisonnement avancé")
st.markdown("Un assistant conversationnel basé sur **DialoGPT** + dynamique triadique dissipative + recherche de connaissances.")

# Chargement de la configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Chargement du modèle en cache (pour Streamlit Cloud)
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

# État de session
if "history" not in st.session_state:
    st.session_state.history = []               # messages échangés
    st.session_state.ttu_state = None            # état interne TTU
    st.session_state.traj = []                   # trajectoire TTU
    st.session_state.knowledge_enabled = config['knowledge']['enabled']

# Barre latérale
st.sidebar.header("⚙️ Paramètres de génération")
temperature = st.sidebar.slider("Température", 0.1, 2.0, config['model']['temperature'], 0.1)
max_new_tokens = st.sidebar.slider("Max nouveaux tokens", 50, 500, 150, 10)
repetition_penalty = st.sidebar.slider("Pénalité de répétition", 1.0, 2.0, config['model']['repetition_penalty'], 0.1)
knowledge_enabled = st.sidebar.checkbox("Activer recherche Wikipédia", value=st.session_state.knowledge_enabled)
st.session_state.knowledge_enabled = knowledge_enabled

mode = st.sidebar.selectbox("Mode", ["Standard", "Dissipation active", "Silence dissipatif", "Exploration"])
if mode == "Dissipation active":
    st.sidebar.info("Créativité maximale")
elif mode == "Silence dissipatif":
    st.sidebar.info("Stabilité")
elif mode == "Exploration":
    st.sidebar.info("Exploration")

if st.sidebar.button("🗑️ Nouvelle conversation"):
    st.session_state.history = []
    st.session_state.ttu_state = None
    st.session_state.traj = []
    st.rerun()

# Interface principale : deux colonnes
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")

    # Afficher l'historique des messages
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Zone de saisie
    prompt = st.chat_input("Posez votre question...")
    if prompt:
        # Afficher le message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})

        # Recherche de connaissances (si activée)
        knowledge = None
        if knowledge_enabled:
            with st.spinner("Recherche de connaissances..."):
                knowledge = search_wikipedia(prompt, max_sentences=config['knowledge']['max_summary_sentences'])
            if knowledge:
                st.info(f"Contexte trouvé : {knowledge}")

        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                response, new_state, traj = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    ttu_state=st.session_state.ttu_state,
                    knowledge=knowledge
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
        last = st.session_state.traj[-1][0]
        st.metric("Cohérence (ϕ_C)", f"{last[0]:.3f}")
        st.metric("Dissipation (ϕ_D)", f"{last[1]:.3f}")
        st.metric("Mémoire (ϕ_M)", f"{last[2]:.3f}")
    else:
        st.info("Posez une question pour voir la trajectoire.")

    # Boutons d'exemples
    st.subheader("Exemples de questions")
    if st.button("Hypothèse de Riemann"):
        st.session_state.history.append({"role": "user", "content": "Explique l'hypothèse de Riemann"})
        st.rerun()
    if st.button("Théorème de Fermat"):
        st.session_state.history.append({"role": "user", "content": "Qu'est-ce que le dernier théorème de Fermat ?"})
        st.rerun()
    if st.button("Qu'est-ce que la beauté ?"):
        st.session_state.history.append({"role": "user", "content": "Qu'est-ce que la beauté selon les philosophes ?"})
        st.rerun()
    if st.button("Savoir ou croyances"):
        st.session_state.history.append({"role": "user", "content": "Savoir ou croyances. Épilogue"})
        st.rerun()

# Pied de page
st.markdown("---")
st.markdown("**TTU-MC³** — Théorie Triadique Unifiée — [Documentation](https://github.com/votre_nom/ttu-ai-advanced)")