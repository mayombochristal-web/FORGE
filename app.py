import streamlit as st
import torch
import numpy as np
import yaml
import os
import plotly.graph_objects as go
from ttu_model import TTULanguageModel
from utils import load_text, plot_trajectory_3d

# Configuration
st.set_page_config(page_title="TTU Language Model", layout="wide")
st.title("🧠 Modèle de Langage TTU-MC³")
st.markdown("Explorez un modèle de langage basé sur la dynamique triadique dissipative.")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Charger un corpus texte", type=['txt'])
if uploaded_file is not None:
    # Sauvegarder temporairement
    with open("temp_corpus.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    corpus_path = "temp_corpus.txt"
else:
    # Utiliser un corpus par défaut (petit échantillon)
    default_text = "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer."
    with open("default.txt", "w") as f:
        f.write(default_text)
    corpus_path = "default.txt"

# Hyperparamètres
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

seq_length = st.sidebar.slider("Longueur de séquence", 10, 200, config['training']['seq_length'])
batch_size = st.sidebar.slider("Taille de batch", 8, 256, config['training']['batch_size'])
lr = st.sidebar.number_input("Taux d'apprentissage", 1e-5, 1e-2, config['training']['learning_rate'], format="%.5f")
epochs = st.sidebar.number_input("Nombre d'époques", 1, 100, config['training']['epochs'])
device = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

# Bouton pour charger/entraîner
if st.sidebar.button("Charger et entraîner"):
    with st.spinner("Chargement du corpus..."):
        dataloader, vocab_size, idx2char, char2idx = load_text(
            corpus_path, seq_length, batch_size, device
        )
        st.session_state['vocab_size'] = vocab_size
        st.session_state['idx2char'] = idx2char
        st.session_state['char2idx'] = char2idx
        st.session_state['dataloader'] = dataloader

    # Initialiser le modèle
    model = TTULanguageModel(
        vocab_size=vocab_size,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dt=config['model']['dt']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    st.session_state['model'] = model
    st.session_state['optimizer'] = optimizer
    st.session_state['criterion'] = criterion
    st.session_state['loss_history'] = []
    st.session_state['dissipation_history'] = []

# Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📈 Entraînement")
    if 'model' in st.session_state:
        if st.button("Lancer une époque"):
            model = st.session_state['model']
            dataloader = st.session_state['dataloader']
            optimizer = st.session_state['optimizer']
            criterion = st.session_state['criterion']

            model.train()
            total_loss = 0
            total_diss = 0
            num_batches = 0
            progress = st.progress(0)
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, traj = model(x)
                loss = criterion(logits.view(-1, st.session_state['vocab_size']), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # Moyenne de phi_d sur la dernière couche pour la dissipation
                last_layer_states = traj[f'layer_{model.num_layers-1}']
                diss_mean = np.mean([s[:,1].mean().item() for s in last_layer_states])
                total_diss += diss_mean
                num_batches += 1
                progress.progress((batch_idx+1)/len(dataloader))

            avg_loss = total_loss / num_batches
            avg_diss = total_diss / num_batches
            st.session_state['loss_history'].append(avg_loss)
            st.session_state['dissipation_history'].append(avg_diss)
            st.success(f"Époque terminée - Loss: {avg_loss:.4f} - Dissipation: {avg_diss:.4f}")

        # Afficher les courbes
        if st.session_state['loss_history']:
            st.subheader("Courbe de perte")
            st.line_chart(st.session_state['loss_history'])
        if st.session_state['dissipation_history']:
            st.subheader("Dissipation moyenne")
            st.line_chart(st.session_state['dissipation_history'])

    else:
        st.info("Chargez un corpus et lancez l'entraînement.")

with col2:
    st.header("🎛️ Visualisation et Génération")
    if 'model' in st.session_state:
        # Visualisation de la trajectoire sur un échantillon
        if st.button("Visualiser une trajectoire"):
            model = st.session_state['model']
            dataloader = st.session_state['dataloader']
            sample_batch, _ = next(iter(dataloader))
            sample_batch = sample_batch[:1].to(device)  # un seul exemple
            with torch.no_grad():
                _, traj = model(sample_batch)
            fig = plot_trajectory_3d(traj, layer=0)
            st.plotly_chart(fig, use_container_width=True)

        # Génération de texte
        st.subheader("Génération de texte")
        prompt = st.text_input("Prompt initial", value="To be")
        max_new = st.slider("Nombre de tokens à générer", 10, 500, 100)
        temperature = st.slider("Température", 0.1, 2.0, 1.0)

        if st.button("Générer"):
            model = st.session_state['model']
            char2idx = st.session_state['char2idx']
            idx2char = st.session_state['idx2char']
            # Convertir le prompt en indices
            start_ids = [char2idx.get(c, 0) for c in prompt]
            start_tensor = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(device)
            generated_ids = model.generate(start_tensor, max_new, temperature)
            generated_chars = [idx2char[i] for i in generated_ids]
            output = prompt + ''.join(generated_chars)
            st.text_area("Texte généré", output, height=200)

    else:
        st.info("Entraînez d'abord un modèle.")

# Pied de page
st.markdown("---")
st.markdown("**Théorie Triadique Unifiée (TTU-MC³)** — Démonstration interactive par [Votre Nom]")