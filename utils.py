import torch
import numpy as np
import plotly.graph_objects as go
import requests
import os

def load_text(filepath, seq_length, batch_size):
    """Charge un fichier texte et crée un DataLoader."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = [char2idx[ch] for ch in text]
    vocab_size = len(chars)

    sequences = []
    targets = []
    for i in range(0, len(data) - seq_length, seq_length):
        seq = data[i:i+seq_length]
        target = data[i+1:i+seq_length+1]
        if len(seq) == seq_length:
            sequences.append(seq)
            targets.append(target)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, vocab_size, idx2char, char2idx

def plot_trajectory_3d(traj):
    """Affiche la trajectoire 3D d'une cellule."""
    points = np.array(traj).squeeze(1)  # (seq_len, 3)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='lines+markers',
        marker=dict(size=2, color=np.arange(len(points)), colorscale='Viridis'),
        line=dict(color='darkblue', width=2)
    )])
    fig.update_layout(
        title="Trajectoire TTU - État interne",
        scene=dict(
            xaxis_title='Phi_C (Cohérence)',
            yaxis_title='Phi_D (Dissipation)',
            zaxis_title='Phi_M (Mémoire)'
        )
    )
    return fig

def download_corpus(name="shakespeare"):
    os.makedirs("data", exist_ok=True)
    urls = {
        'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'tiny_shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'wikitext': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt'
    }
    url = urls.get(name, urls['shakespeare'])
    dest = os.path.join('data', f"{name}.txt")
    if not os.path.exists(dest):
        print(f"Téléchargement de {url}...")
        r = requests.get(url)
        with open(dest, 'wb') as f:
            f.write(r.content)
    return dest