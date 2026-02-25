import streamlit as st
import torch
import numpy as np
import yaml
import os
from pathlib import Path
from ttu_model import TTULanguageModel
from utils import load_text, plot_trajectory_3d, download_corpus
import plotly.graph_objects as go
import time

# ---------- Configuration de la page ----------
st.set_page_config(page_title="TTU-MC³ Chatbot", layout="wide")
st.title("🧠💬 Chatbot TTU-MC³")
st.markdown("Un assistant conversationnel basé sur la dynamique triadique dissipative.")

# ---------- Chargement de la config ----------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------- Initialisation des dossiers ----------
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ---------- Classe pour gérer la conversation ----------
class Conversation:
    def __init__(self, model, device, mode_params):
        self.model = model
        self.device = device
        self.mode_params = mode_params
        self.reset()

    def reset(self):
        self.history = []          # liste de {"role": "user"/"assistant", "content": str}
        self.states = [None] * self.model.num_layers
        self.trajectory = None

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_state(self):
        return self.states

    def set_state(self, states):
        self.states = states

# ---------- Barre latérale ----------
st.sidebar.header("⚙️ Paramètres")

# Mode de génération
mode = st.sidebar.selectbox(
    "Mode de dissipation",
    ["Dissipation Complexe Active", "Silence Dissipatif", "Mode Bruit", "Mode Exploration"]
)

# Paramètres ajustables selon le mode
if mode == "Dissipation Complexe Active":
    gamma = st.sidebar.slider("gamma (gain)", 0.5, 2.5, 1.35, 0.05)
    mu = st.sidebar.slider("mu (friction)", 0.1, 2.0, 0.75, 0.05)
    noise = st.sidebar.slider("bruit", 0.0, 0.5, 0.05, 0.01)
elif mode == "Silence Dissipatif":
    gamma = st.sidebar.slider("gamma", 0.5, 2.0, 0.8, 0.05)
    mu = st.sidebar.slider("mu", 0.1, 1.0, 0.3, 0.05)
    noise = 0.0
elif mode == "Mode Bruit":
    gamma = st.sidebar.slider("gamma", 0.5, 2.0, 1.0, 0.05)
    mu = st.sidebar.slider("mu", 0.1, 1.5, 0.5, 0.05)
    noise = st.sidebar.slider("bruit", 0.0, 1.0, 0.2, 0.05)
elif mode == "Mode Exploration":
    gamma = st.sidebar.slider("gamma", 1.0, 3.0, 2.0, 0.1)
    mu = st.sidebar.slider("mu", 0.5, 2.5, 1.2, 0.1)
    noise = st.sidebar.slider("bruit", 0.0, 0.8, 0.3, 0.05)

temperature = st.sidebar.slider("Température", 0.1, 2.0, 1.0, 0.1)
max_new_tokens = st.sidebar.slider("Longueur max réponse", 20, 300, 150, 10)

# ---------- Gestion des modèles ----------
st.sidebar.header("💾 Modèle")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

# Liste des modèles disponibles
model_files = list(Path("models").glob("*.pt"))
model_names = [f.stem for f in model_files]
selected_model_name = st.sidebar.selectbox("Charger un modèle", ["Aucun"] + model_names)

if selected_model_name != "Aucun":
    load_path = os.path.join("models", f"{selected_model_name}.pt")
    if st.sidebar.button("Charger"):
        checkpoint = torch.load(load_path, map_location=device)
        # Reconstruire le modèle
        model = TTULanguageModel(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['config']['model']['embed_dim'],
            hidden_dim=checkpoint['config']['model']['hidden_dim'],
            num_layers=checkpoint['config']['model']['num_layers'],
            dt=checkpoint['config']['model']['dt']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.session_state['model'] = model
        st.session_state['idx2char'] = checkpoint['idx2char']
        st.session_state['char2idx'] = checkpoint['char2idx']
        st.session_state['vocab_size'] = checkpoint['vocab_size']
        st.session_state['config'] = checkpoint['config']
        st.sidebar.success(f"Modèle {selected_model_name} chargé.")

# Sauvegarde du modèle actuel
model_name_save = st.sidebar.text_input("Nom pour sauvegarde", value="ttu_model")
if st.sidebar.button("Sauvegarder le modèle"):
    if 'model' in st.session_state:
        save_path = os.path.join("models", f"{model_name_save}.pt")
        torch.save({
            'model_state_dict': st.session_state['model'].state_dict(),
            'idx2char': st.session_state['idx2char'],
            'char2idx': st.session_state['char2idx'],
            'vocab_size': st.session_state['vocab_size'],
            'config': config
        }, save_path)
        st.sidebar.success(f"Modèle sauvegardé sous {save_path}")
    else:
        st.sidebar.error("Aucun modèle à sauvegarder.")

# ---------- Entraînement ----------
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
epochs = st.sidebar.number_input("Époques", 1, 100, config['training']['epochs'])

if st.sidebar.button("Charger corpus et initialiser modèle"):
    with st.spinner("Chargement du corpus..."):
        dataloader, vocab_size, idx2char, char2idx = load_text(
            corpus_path, seq_length, batch_size, device
        )
        st.session_state['dataloader'] = dataloader
        st.session_state['vocab_size'] = vocab_size
        st.session_state['idx2char'] = idx2char
        st.session_state['char2idx'] = char2idx

    # Initialiser le modèle
    model = TTULanguageModel(
        vocab_size=vocab_size,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dt=config['model']['dt']
    ).to(device)
    st.session_state['model'] = model
    st.session_state['optimizer'] = torch.optim.Adam(model.parameters(), lr=lr)
    st.session_state['criterion'] = torch.nn.CrossEntropyLoss()
    st.session_state['loss_history'] = []
    st.session_state['dissipation_history'] = []
    st.success("Modèle initialisé. Vous pouvez maintenant lancer l'entraînement.")

if 'model' in st.session_state and st.sidebar.button("Lancer une époque d'entraînement"):
    model = st.session_state['model']
    dataloader = st.session_state['dataloader']
    optimizer = st.session_state['optimizer']
    criterion = st.session_state['criterion']

    model.train()
    total_loss = 0
    total_diss = 0
    num_batches = 0
    progress = st.sidebar.progress(0)
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, traj = model(x)
        loss = criterion(logits.view(-1, st.session_state['vocab_size']), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Dissipation moyenne sur la dernière couche
        last_layer_states = traj[f'layer_{model.num_layers-1}']
        diss_mean = np.mean([s[:,1].mean().item() for s in last_layer_states])
        total_diss += diss_mean
        num_batches += 1
        progress.progress((batch_idx+1)/len(dataloader))

    avg_loss = total_loss / num_batches
    avg_diss = total_diss / num_batches
    st.session_state['loss_history'].append(avg_loss)
    st.session_state['dissipation_history'].append(avg_diss)
    st.sidebar.success(f"Loss: {avg_loss:.4f} | Dissipation: {avg_diss:.4f}")

# ---------- Interface principale ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")

    if 'model' in st.session_state:
        # Initialiser la conversation si nécessaire
        if 'conversation' not in st.session_state:
            mode_params = {'gamma': gamma, 'mu': mu, 'noise': noise}
            st.session_state['conversation'] = Conversation(
                st.session_state['model'],
                device,
                mode_params
            )

        conv = st.session_state['conversation']

        # Afficher l'historique
        for msg in conv.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Saisie utilisateur
        prompt = st.chat_input("Votre message...")
        if prompt:
            # Afficher le message utilisateur
            with st.chat_message("user"):
                st.markdown(prompt)
            conv.add_message("user", prompt)

            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    # Tokeniser le prompt
                    char2idx = st.session_state['char2idx']
                    idx2char = st.session_state['idx2char']
                    input_ids = [char2idx.get(c, 0) for c in prompt]
                    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

                    # Appliquer les paramètres du mode au modèle
                    model = st.session_state['model']
                    for cell in model.cells:
                        cell.gamma.data = torch.tensor(gamma, device=device)

                    # Faire évoluer le modèle sur le prompt (mise à jour de l'état)
                    with torch.no_grad():
                        states = conv.get_state()
                        for i in range(input_tensor.size(1)):
                            x = input_tensor[:, i:i+1]
                            embeds = model.embedding(x).squeeze(1)
                            new_states = []
                            for j, cell in enumerate(model.cells):
                                state_j = states[j] if states[j] is not None else None
                                new_state = cell(embeds, state_j)
                                new_states.append(new_state)
                            states = new_states
                        conv.set_state(states)

                        # Générer la réponse
                        generated_ids, new_states = model.generate(
                            start_ids=input_tensor,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            states=states
                        )
                        response_text = ''.join([idx2char[i] for i in generated_ids])
                        conv.set_state(new_states)

                    st.markdown(response_text)
                    conv.add_message("assistant", response_text)

            # Re-run pour afficher le nouveau message
            st.rerun()

        # Bouton pour reset
        if st.button("🗑️ Nouvelle conversation"):
            conv.reset()
            st.rerun()

    else:
        st.info("Veuillez charger ou initialiser un modèle (via la barre latérale).")

with col2:
    st.header("📈 Visualisation")

    if 'model' in st.session_state and 'conversation' in st.session_state:
        if st.button("Afficher une trajectoire exemple"):
            model = st.session_state['model']
            if 'dataloader' in st.session_state:
                sample_batch, _ = next(iter(st.session_state['dataloader']))
                sample_batch = sample_batch[:1].to(device)
                with torch.no_grad():
                    _, traj = model(sample_batch)
                fig = plot_trajectory_3d(traj, layer=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucun dataloader disponible.")
    else:
        st.info("Visualisation disponible après chargement du modèle.")

    # Afficher les courbes d'entraînement si disponibles
    if 'loss_history' in st.session_state and st.session_state['loss_history']:
        st.subheader("Courbe de perte")
        st.line_chart(st.session_state['loss_history'])
    if 'dissipation_history' in st.session_state and st.session_state['dissipation_history']:
        st.subheader("Dissipation moyenne")
        st.line_chart(st.session_state['dissipation_history'])

# ---------- Pied de page ----------
st.markdown("---")
st.markdown("**TTU-MC³** — Théorie Triadique Unifiée — [Documentation](https://github.com/votre_nom/ttu-chatbot)")