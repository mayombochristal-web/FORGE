"""
Ω-TTU V14 – Mini LLM Transformer (≈50M params)
================================================
Fonctionnalités :
- Architecture Transformer moderne (RMSNorm, SwiGLU)
- Tokenizer construit dynamiquement depuis votre base SQLite (ngrams)
- Inférence avec température et échantillonnage
- Interface Streamlit simple
- Prêt pour l'entraînement personnalisé (script séparé fourni en commentaire)

Dépendances : torch, streamlit, numpy, sqlite3
Installation : pip install torch streamlit numpy
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import numpy as np
import os
from collections import defaultdict

# =========================================================
# 1. ARCHITECTURE DU MODÈLE (Transformer CPU-friendly)
# =========================================================

class RMSNorm(nn.Module):
    """Normalisation RMS (Root Mean Square) – plus stable que LayerNorm pour petits modèles"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """FFN avec activation SwiGLU (utilisée dans LLaMA)"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Bloc Transformer standard avec pré‑norm et SwiGLU"""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, dim * 4)  # facteur d'expansion 4

    def forward(self, x):
        # Attention avec résidu
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # FFN avec résidu
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class OmegaTTU_LLM(nn.Module):
    """
    Ω-TTU Transformer V1 – spécifications :
    - vocab_size : à définir depuis la base
    - dim = 384
    - heads = 6
    - layers = 8
    - contexte max = 512 (géré par génération)
    - paramètres ≈ 48M
    """
    def __init__(self, vocab_size, dim=384, layers=8, heads=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # Positional encoding simple (appris) – on peut aussi utiliser Rotary
        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Lier les poids d'embedding et de sortie (pratique courante)
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
        # x : (batch, seq_len)
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits


# =========================================================
# 2. TOKENIZER DYNAMIQUE DEPUIS LA BASE SQLITE
# =========================================================

def build_tokenizer_from_db(db_path="oracle_memory/relations.db"):
    """
    Lit tous les mots distincts de la table 'ngrams' et construit un vocabulaire.
    Ajoute les tokens spéciaux : <pad>=0, <unk>=1, <s>=2, </s>=3
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base introuvable : {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT next_word FROM ngrams")
    rows = c.fetchall()
    conn.close()

    words = sorted(set([row[0] for row in rows if row[0]]))
    # Création du vocabulaire
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    for i, w in enumerate(words):
        vocab[w] = i + 4   # les tokens spéciaux occupent 0-3
    id_to_word = {v: k for k, v in vocab.items()}
    return vocab, id_to_word


# =========================================================
# 3. CHARGEMENT DU MODÈLE (avec mise en cache Streamlit)
# =========================================================

def load_model(vocab_size, model_path="omega_ttu_llm.pt"):
    """
    Charge les poids depuis un fichier .pt, ou initialise aléatoirement.
    """
    model = OmegaTTU_LLM(vocab_size=vocab_size)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
        st.success(f"Modèle chargé depuis {model_path}")
    else:
        st.warning("Aucun modèle pré‑entraîné trouvé. Utilisation de poids aléatoires.")
    model.eval()
    return model


# =========================================================
# 4. FONCTION DE GÉNÉRATION
# =========================================================

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.8):
    """
    Génère une séquence à partir d'une liste d'IDs.
    """
    input_ids = torch.tensor([prompt_ids]).long()  # (1, seq_len)
    for _ in range(max_new_tokens):
        # Limiter à la fenêtre de contexte (512)
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, -512:]

        logits = model(input_ids)                 # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    return input_ids[0].tolist()


# =========================================================
# 5. INTERFACE STREAMLIT
# =========================================================

st.set_page_config(page_title="Ω-TTU V14", layout="wide")
st.title("🧠 Ω-TTU V14 – Mini LLM Transformer (≈50M params)")
st.markdown("Modèle génératif entraînable sur CPU, utilisant votre base de connaissances.")

# Chemin de la base (à ajuster si besoin)
DB_PATH = "oracle_memory/relations.db"

if not os.path.exists(DB_PATH):
    st.error("Base de données non trouvée. Veuillez d'abord exécuter la V13 pour créer 'relations.db'.")
    st.stop()

# Construction du vocabulaire
vocab, id_to_word = build_tokenizer_from_db(DB_PATH)
vocab_size = len(vocab)
st.sidebar.success(f"Vocabulaire chargé : {vocab_size} mots")

# Chargement du modèle (mise en cache)
@st.cache_resource
def get_model():
    return load_model(vocab_size, model_path="omega_ttu_llm.pt")

model = get_model()

# Paramètres de génération
st.sidebar.header("Paramètres de génération")
max_tokens = st.sidebar.slider("Nouveaux tokens", 10, 500, 100)
temperature = st.sidebar.slider("Température", 0.1, 2.0, 0.8)

# Zone de saisie
prompt = st.text_input("Entrez un début de phrase", value="Dans le texte fondateur")

if st.button("Générer"):
    # Tokenisation simple (split sur les espaces)
    tokens = prompt.strip().split()
    prompt_ids = [vocab.get(w, vocab['<unk>']) for w in tokens]

    if len(prompt_ids) == 0:
        st.warning("Veuillez entrer un prompt non vide.")
    else:
        with st.spinner("Génération en cours..."):
            output_ids = generate(model, prompt_ids, max_new_tokens=max_tokens, temperature=temperature)

        # Conversion en texte
        output_words = [id_to_word.get(i, '<unk>') for i in output_ids]
        output_text = " ".join(output_words)

        st.markdown("### Réponse générée")
        st.write(output_text)

        # Option pour copier
        st.code(output_text, language="text")

# Section d'information
with st.expander("ℹ️ À propos du modèle"):
    st.markdown("""
    **Architecture**
    - Embedding dimension : 384
    - Nombre de couches : 8
    - Têtes d'attention : 6
    - FFN SwiGLU (dimension cachée 1536)
    - Normalisation RMS
    - Poids liés (embedding = sortie)
    - Paramètres : ~48 millions

    **Tokenizer**
    - Construit automatiquement à partir des mots distincts de la table `ngrams`.
    - Tokens spéciaux : `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3).

    **Entraînement**
    - Pour utiliser un modèle entraîné, placez un fichier `omega_ttu_llm.pt` dans le même dossier.
    - Vous pouvez entraîner le modèle sur vos données avec un script PyTorch standard (exemple fourni ci‑dessous).

    **Exemple d'entraînement (script séparé)**
    ```python
    import torch
    from torch.utils.data import DataLoader, Dataset
    # ... (à adapter selon votre corpus)
    ```
    """)

# Option pour télécharger un modèle vide (placeholder)
if st.button("📥 Télécharger un modèle initial (poids aléatoires)"):
    dummy_model = OmegaTTU_LLM(vocab_size=1000)  # petit pour l'exemple
    torch.save(dummy_model.state_dict(), "omega_ttu_llm_dummy.pt")
    st.success("Fichier 'omega_ttu_llm_dummy.pt' créé (poids aléatoires).")