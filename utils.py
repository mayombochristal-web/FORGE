import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def load_text(filepath, seq_length, batch_size, device='cpu'):
    """Charge un fichier texte et crée un DataLoader pour l'entraînement."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = [char2idx[ch] for ch in text]
    vocab_size = len(chars)

    # Découpage en séquences
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

def plot_trajectory_3d(traj, layer=0):
    """Affiche la trajectoire d'une cellule dans l'espace 3D."""
    # traj: liste de (batch, 3) pour chaque pas de temps, batch=1 ici
    points = np.array(traj[f'layer_{layer}']).squeeze(1)  # (seq_len, 3)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='lines+markers',
        marker=dict(size=2, color=np.arange(len(points)), colorscale='Viridis'),
        line=dict(color='darkblue', width=2)
    )])
    fig.update_layout(
        title=f"Trajectoire TTU - Couche {layer}",
        scene=dict(
            xaxis_title='Phi_C (Cohérence)',
            yaxis_title='Phi_D (Dissipation)',
            zaxis_title='Phi_M (Mémoire)'
        )
    )
    return fig

def plot_dissipation(dissipation_history):
    """Trace l'évolution de la dissipation au cours de l'entraînement."""
    fig, ax = plt.subplots()
    ax.plot(dissipation_history, label='Dissipation moyenne')
    ax.set_xlabel('Étape')
    ax.set_ylabel('Phi_D')
    ax.set_title('Évolution de la dissipation')
    ax.legend()
    return fig

def compute_lyapunov(cell, state, impulse, dt=0.1):
    """Calcule une approximation du spectre de Lyapunov via le Jacobien."""
    # À implémenter si besoin
    pass