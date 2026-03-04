import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Configuration (adaptez ici)
DB_PATH = 'relations.db'
VOCAB_SIZE = 50000  # Taille approx. vocabulaire depuis DB
EMBED_DIM = 512
HEADS = 8
LAYERS = 6
FF_DIM = 2048
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
SAVE_PATH = 'omega_ttu_llm.pt'

# 1. Extraire vocabulaire depuis DB
def build_vocab(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT word FROM ngrams LIMIT ?;", (VOCAB_SIZE,))
    words = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    for i, w in enumerate(words, 4):
        vocab[w] = i
    return vocab, {v: k for k, v in vocab.items()}

vocab, id2word = build_vocab(DB_PATH)
vocab_size = len(vocab)

# 2. Dataset : charger ngrams et créer séquences auto-régressives
class NgramsDataset(Dataset):
    def __init__(self, db_path, vocab, max_len=MAX_SEQ_LEN):
        self.vocab = vocab
        self.max_len = max_len
        self.sequences = self._load_sequences(db_path)
    
    def _load_sequences(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT word FROM ngrams ORDER BY RANDOM() LIMIT 100000;")  # Échantillon aléatoire
        words = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Construire séquences : mots concaténés en phrases simulées
        seqs = []
        for i in range(0, len(words) - self.max_len, 1):
            seq = words[i:i + self.max_len]
            seqs.append([self.vocab.get(w, 1) for w in seq])  # <unk> si absent
        return seqs
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1])
        labels = torch.tensor(seq[1:])
        return input_ids, labels

dataset = NgramsDataset(DB_PATH, vocab)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / norm * self.scale

# 4. SwiGLU
class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

# 5. Mini-LLM Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm2 = RMSNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim * 2),
            SwiGLU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + attn_out
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        return x + ff_out

# 6. Modèle Complet (weight tying)
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, layers, ff_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, ff_dim) for _ in range(layers)])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = self.embed  # Weight tying
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

model = MiniLLM(vocab_size, EMBED_DIM, HEADS, LAYERS, FF_DIM)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad>

# 7. Boucle d'entraînement
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    total_loss = 0
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"Modèle sauvegardé : {SAVE_PATH}")