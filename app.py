import streamlit as st
import torch
import yaml
import os
from pathlib import Path
from ttu_model import TTULanguageModel
from utils import load_text, plot_trajectory_3d, download_corpus
import numpy as np

# Configuration de la page
st.set_page_config(page_title="TTU-MC³ IA Autonome", layout="wide")
st.title("🧠🤖 TTU-MC³ IA Autonome")
st.markdown("Un générateur de texte autonome basé sur la dynamique triadique dissipative (sans modèle pré-entraîné).")

# Chargement config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dossiers
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.ttu_states = None
    st.session_state.traj = []
    st.session_state.model_loaded = False
    st.session_state.vocab_size = 0
    st.session_state.idx2char = {}
    st.session_state.char2idx = {}
    st.session_state.loss_history = []
    st.session_state.dissipation_history = []

# Barre latérale
st.sidebar.header("⚙️ Paramètres")

# Mode de génération
temperature = st.sidebar.slider("Température", 0.1, 2.0, 1.0, 0.1)
max_new_tokens = st.sidebar.slider("Max nouveaux tokens", 20, 500, 150, 10)

# Choix du device
device = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

# Gestion des modèles sauvegardés
model_files = list(Path("models").glob("*.pt"))
model_names = [f.stem for f in model_files]
selected_model = st.sidebar.selectbox("Charger un modèle", ["Aucun"] + model_names)

if selected_model != "Aucun" and st.sidebar.button("Charger"):
    load_path = os.path.join("models", f"{selected_model}.pt")
    checkpoint = torch.load(load_path, map_location=device)
    # Reconstruire le modèle
    model = TTULanguageModel(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dt=config['model']['dt']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    st.session_state.model = model
    st.session_state.idx2char = checkpoint['idx2char']
    st.session_state.char2idx = checkpoint['char2idx']
    st.session_state.vocab_size = checkpoint['vocab_size']
    st.session_state.model_loaded = True
    st.session_state.history = []
    st.session_state.ttu_states = None
    st.sidebar.success(f"Modèle {selected_model} chargé.")

# Sauvegarde
model_name_save = st.sidebar.text_input("Nom pour sauvegarde", value="ttu_model")
if st.sidebar.button("Sauvegarder le modèle"):
    if st.session_state.model_loaded:
        save_path = os.path.join("models", f"{model_name_save}.pt")
        torch.save({
            'model_state_dict': st.session_state.model.state_dict(),
            'idx2char': st.session_state.idx2char,
            'char2idx': st.session_state.char2idx,
            'vocab_size': st.session_state.vocab_size
        }, save_path)
        st.sidebar.success(f"Modèle sauvegardé sous {save_path}")
    else:
        st.sidebar.error("Aucun modèle à sauvegarder.")

# Entraînement
st.sidebar.header("🎓 Entraînement")
corpus_option = st.sidebar.radio("Corpus", ["Défaut (Shakespeare)", "Télécharger", "Upload"])
if corpus_option == "Défaut (Shakespeare)":
    corpus_path = download_corpus("shakespeare")
elif corpus_option == "Télécharger":
    corpus_name = st.sidebar.text_input("Nom du corpus (shakespeare, wikitext)", "shakespeare")
    if st.sidebar.button("Télécharger"):
        corpus_path = download_corpus(corpus_name)
        st.sidebar.success(f"Corpus {corpus_name} téléchargé.")
    else:
        corpus_path = download_corpus("shakespeare")
else:
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier texte", type=['txt'])
    if uploaded_file:
        corpus_path = os.path.join("data", "uploaded.txt")
        with open(corpus_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("Fichier uploadé.")
    else:
        corpus_path = download_corpus("shakespeare")

seq_length = st.sidebar.slider("Longueur séquence", 20, 200, config['training']['seq_length'])
batch_size = st.sidebar.slider("Batch size", 8, 256, config['training']['batch_size'])
lr = st.sidebar.number_input("Learning rate", 1e-5, 1e-2, config['training']['learning_rate'], format="%.5f")
epochs_per_click = st.sidebar.number_input("Époques par clic", 1, 20, config['training']['epochs_per_click'])

if st.sidebar.button("Initialiser le modèle"):
    with st.spinner("Chargement du corpus..."):
        dataloader, vocab_size, idx2char, char2idx = load_text(
            corpus_path, seq_length, batch_size
        )
        st.session_state.dataloader = dataloader
        st.session_state.vocab_size = vocab_size
        st.session_state.idx2char = idx2char
        st.session_state.char2idx = char2idx

    # Initialiser le modèle
    model = TTULanguageModel(
        vocab_size=vocab_size,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dt=config['model']['dt']
    ).to(device)
    st.session_state.model = model
    st.session_state.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    st.session_state.criterion = torch.nn.CrossEntropyLoss()
    st.session_state.loss_history = []
    st.session_state.dissipation_history = []
    st.session_state.model_loaded = True
    st.success("Modèle initialisé. Vous pouvez maintenant lancer l'entraînement.")

if st.session_state.model_loaded and st.sidebar.button("Lancer entraînement"):
    model = st.session_state.model
    dataloader = st.session_state.dataloader
    optimizer = st.session_state.optimizer
    criterion = st.session_state.criterion

    model.train()
    total_loss = 0
    total_diss = 0
    num_batches = 0
    progress = st.sidebar.progress(0)
    for epoch in range(epochs_per_click):
        epoch_loss = 0
        epoch_diss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, states, traj = model(x)
            loss = criterion(logits.view(-1, st.session_state.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Dissipation moyenne (dernière couche)
            last_layer = traj[f'layer_{model.num_layers-1}']
            diss_mean = np.mean([s[:,1].mean().item() for s in last_layer])
            epoch_diss += diss_mean
            num_batches += 1
            progress.progress((batch_idx+1)/len(dataloader))
        total_loss += epoch_loss / len(dataloader)
        total_diss += epoch_diss / len(dataloader)

    avg_loss = total_loss / epochs_per_click
    avg_diss = total_diss / epochs_per_click
    st.session_state.loss_history.append(avg_loss)
    st.session_state.dissipation_history.append(avg_diss)
    st.sidebar.success(f"Loss: {avg_loss:.4f} | Dissipation: {avg_diss:.4f}")

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")

    if not st.session_state.model_loaded:
        st.info("Veuillez initialiser ou charger un modèle via la barre latérale.")
    else:
        # Afficher l'historique
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Saisie utilisateur
        prompt = st.chat_input("Votre message...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.history.append({"role": "user", "content": prompt})

            # Tokeniser le prompt
            input_ids = [st.session_state.char2idx.get(c, 0) for c in prompt]
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Génération..."):
                    generated_ids, new_states, traj = st.session_state.model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        states=st.session_state.ttu_states
                    )
                    response_text = ''.join([st.session_state.idx2char[i] for i in generated_ids])
                    st.markdown(response_text)
                    st.session_state.history.append({"role": "assistant", "content": response_text})
                    st.session_state.ttu_states = new_states
                    st.session_state.traj = traj

        # Bouton reset
        if st.button("🗑️ Nouvelle conversation"):
            st.session_state.history = []
            st.session_state.ttu_states = None
            st.session_state.traj = []
            st.rerun()

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

    # Courbes d'entraînement
    if st.session_state.loss_history:
        st.subheader("Courbe de perte")
        st.line_chart(st.session_state.loss_history)
    if st.session_state.dissipation_history:
        st.subheader("Dissipation moyenne")
        st.line_chart(st.session_state.dissipation_history)

st.markdown("---")
st.markdown("**TTU-MC³** — Générateur autonome — [Documentation](https://github.com/votre_nom/ttu-autonomous-ai)")