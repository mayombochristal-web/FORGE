import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, datetime, time, base64, math, sqlite3
from collections import Counter
import requests
from scipy.signal import stft
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ORACLE Ω-TTU V14",
    layout="wide",
    page_icon="🧬"
)

MEM_DIR = "oracle_memory"
os.makedirs(MEM_DIR, exist_ok=True)

DB_PATH = os.path.join(MEM_DIR, "relations.db")

# =========================================================
# CHARGEMENT MODELE EMBEDDING
# =========================================================

@st.cache_resource
def load_embedder():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model

embed_model = load_embedder()

# =========================================================
# DATABASE
# =========================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS ngrams(
            context TEXT,
            next_word TEXT,
            weight REAL,
            lyapunov REAL,
            PRIMARY KEY(context, next_word)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================================================
# CORE TTU-MC³
# =========================================================

def get_triadic_metrics(text):
    """
    Calcule ΦM, ΦC, ΦD selon la méthodologie TTU-MC³
    """

    embeddings = embed_model.encode([text])[0]

    # ΦM : mémoire
    phi_m = np.linalg.norm(embeddings) / 10.0

    # ΦC : cohérence spectrale
    phi_c = np.mean(np.abs(np.fft.fft(embeddings))) / 5.0

    # ΦD : dissipation
    phi_d = 1.0 - np.std(embeddings)

    norm = math.sqrt(phi_m**2 + phi_c**2 + phi_d**2) + 1e-9

    return (
        phi_m / norm,
        phi_c / norm,
        phi_d / norm
    )


def triadic_entropy(text, beta=0.5):
    """
    Entropie triadique HΦ
    """

    phi_m, _, _ = get_triadic_metrics(text)

    counts = Counter(text)

    probs = [
        c / len(text)
        for c in counts.values()
    ]

    shannon = -sum(
        p * math.log2(p)
        for p in probs
    )

    h_phi = shannon - beta * math.log(max(phi_m, 1e-5))

    return h_phi


# =========================================================
# LYAPUNOV
# =========================================================

def get_lyapunov_scale(weight):

    if weight > 2.0:
        return "Stable (λ < 0) — Logique structurée"

    if weight > 1.0:
        return "Neutre (λ ≈ 0) — Cohérente"

    return "Exploratrice (λ > 0) — Créative"


def check_stability(m, c, d):

    return (m * c * d) > 0.01


# =========================================================
# INTERFACE STREAMLIT
# =========================================================

st.title("🧬 ORACLE Ω-TTU V14")
st.write("Refondation des probabilités : de Kolmogorov vers la Triade Informationnelle")

if "phi" not in st.session_state:

    st.session_state.phi = {
        "m": 0.5,
        "c": 0.5,
        "d": 0.5
    }

col1, col2 = st.columns([1, 1])

# =========================================================
# INPUT
# =========================================================

with col1:

    st.subheader("📥 Encodage informationnel")

    input_text = st.text_area(
        "Entrez un texte multilingue",
        height=150
    )

    if st.button("Transformer en champ informationnel"):

        if input_text:

            m, c, d = get_triadic_metrics(input_text)

            h_phi = triadic_entropy(input_text)

            st.session_state.phi = {
                "m": m,
                "c": c,
                "d": d
            }

            st.metric("Entropie triadique HΦ", round(h_phi, 4))

            if check_stability(m, c, d):

                st.success(
                    f"Système stable — Dissipation D = {round(d,2)}"
                )

            else:

                st.warning(
                    "Dissipation faible — exploration de trajectoires instables"
                )

# =========================================================
# VISUALISATION
# =========================================================

with col2:

    st.subheader("📊 Organisation topologique")

    m = st.session_state.phi["m"]
    c = st.session_state.phi["c"]
    d = st.session_state.phi["d"]

    st.write(
        f"Échelle dynamique : {get_lyapunov_scale(m + c)}"
    )

    chart_data = pd.DataFrame({

        "Composante": [
            "Mémoire ΦM",
            "Cohérence ΦC",
            "Dissipation ΦD"
        ],

        "Intensité": [
            m,
            c,
            d
        ]

    })

    st.bar_chart(
        chart_data,
        x="Composante",
        y="Intensité"
    )

st.divider()

st.caption(
    "ORACLE V14 — Implémentation expérimentale du modèle TTU-MC³"
)