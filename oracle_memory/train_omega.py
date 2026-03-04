"""
Script d'entraînement pour Ω-TTU V14
=====================================
Ce script :
- Charge la base SQLite (relations.db)
- Construit le vocabulaire à partir des mots distincts
- Extrait les séquences d'entraînement (soit depuis la timeline du cortex, soit en parcourant tous les n-grammes)
- Entraîne le modèle Transformer (OmegaTTU_LLM)
- Sauvegarde les poids dans 'omega_ttu_llm.pt'
"""

import sqlite3
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# =========================================================
# 1. Architecture du modèle (identique à celle de l'app)
# =========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, dim * 4)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class OmegaTTU_LLM(nn.Module):
    def __init__(self, vocab_size, dim=384, layers=8, heads=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# =========================================================
# 2. Construction du vocabulaire depuis la base
# =========================================================

def build_vocab_from_db(db_path="oracle_memory/relations.db"):
    """Extrait tous les mots distincts de la table ngrams et retourne vocabulaire et mapping inverse."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base introuvable : {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT next_word FROM ngrams")
    rows = c.fetchall()
    conn.close()

    words = sorted(set([row[0] for row in rows if row[0]]))
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    for i, w in enumerate(words):
        vocab[w] = i + 4
    id_to_word = {v: k for k, v in vocab.items()}
    return vocab, id_to_word


# =========================================================
# 3. Extraction des séquences d'entraînement
# =========================================================

def extract_sequences_from_timeline(db_path="oracle_memory/relations.db", cortex_file="oracle_memory/cortex.json"):
    """
    Récupère la timeline depuis le fichier cortex.json et retourne une liste d'IDs.
    """
    if not os.path.exists(cortex_file):
        raise FileNotFoundError(f"Fichier cortex introuvable : {cortex_file}")
    with open(cortex_file, 'r') as f:
        cortex = json.load(f)
    timeline = cortex.get("timeline", [])
    return timeline

def extract_sequences_from_ngrams(db_path="oracle_memory/relations.db"):
    """
    Parcourt tous les n-grammes pour reconstituer des séquences ordonnées (moins fiable).
    On préfère utiliser la timeline.
    """
    # Alternative: on peut parcourir la table et ordonner par ? Il n'y a pas d'ordre.
    # Donc on utilisera la timeline.
    pass

def build_dataset(vocab, timeline_words, seq_len=128):
    """
    Convertit la liste de mots en une liste d'IDs, puis crée des séquences de longueur seq_len.
    """
    ids = [vocab.get(w, vocab['<unk>']) for w in timeline_words]
    # On ajoute des tokens de début et fin si besoin (optionnel)
    # ids = [vocab['<s>']] + ids + [vocab['</s>']]
    X, Y = [], []
    for i in range(0, len(ids) - seq_len - 1, seq_len//2):  # pas glissant
        x = ids[i:i+seq_len]
        y = ids[i+1:i+seq_len+1]
        X.append(x)
        Y.append(y)
    return X, Y


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


# =========================================================
# 4. Entraînement
# =========================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de {device}")

    db_path = "oracle_memory/relations.db"
    cortex_path = "oracle_memory/cortex.json"

    # 1. Vocabulaire
    vocab, id_to_word = build_vocab_from_db(db_path)
    vocab_size = len(vocab)
    print(f"Taille du vocabulaire : {vocab_size}")

    # 2. Séquence d'entraînement
    timeline_words = extract_sequences_from_timeline(db_path, cortex_path)
    if len(timeline_words) < 200:
        print("Timeline trop courte. Utilisation des mots distincts comme séquence aléatoire (non recommandé).")
        # Fallback: prendre tous les mots distincts et les répéter ?
        words_list = list(vocab.keys())[4:]  # sans les spéciaux
        timeline_words = words_list * 10  # répétition artificielle
    print(f"Nombre de mots dans la timeline : {len(timeline_words)}")

    seq_len = 128
    X, Y = build_dataset(vocab, timeline_words, seq_len)
    dataset = TextDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. Modèle
    model = OmegaTTU_LLM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 4. Boucle d'entraînement
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (batch, seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} terminée, perte moyenne : {avg_loss:.4f}")

    # 5. Sauvegarde
    torch.save(model.state_dict(), "omega_ttu_llm.pt")
    print("Modèle sauvegardé sous 'omega_ttu_llm.pt'")

if __name__ == "__main__":
    train()