import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, datetime, time, base64, math, sqlite3
from collections import Counter, deque
import random
import requests
from scipy.signal import stft
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# =========================================================
# CONFIGURATION & INITIALISATION V14
# =========================================================
st.set_page_config(page_title="ORACLE Ω-TTU V14", layout="wide", page_icon="🧬")
MEM_DIR = "oracle_memory"
os.makedirs(MEM_DIR, exist_ok=True)
DB_PATH = os.path.join(MEM_DIR, "relations.db")

# Chargement du modèle d'embeddings pour l'Espace de Hilbert Triadique
@st.cache_resource
def load_embedder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embed_model = load_embedder()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ngrams 
                 (context TEXT, next_word TEXT, weight REAL, lyapunov REAL, PRIMARY KEY (context, next_word))''')
    conn.commit()
    conn.close()

init_db()

# =========================================================
# CORE TTU-MC³ : ESTIMATEUR D'ENTROPIE TRIADIQUE
# =========================================================
def get_triadic_metrics(text):
    [span_4](start_span)[span_5](start_span)"""Calcule ΦM, ΦC, ΦD selon la méthodologie TTU-MC³[span_4](end_span)[span_5](end_span)"""
    embeddings = embed_model.encode([text])[0]
    
    # ΦM (Mémoire) : Persistance par corrélation (norme de l'embedding)
    phi_m = np.linalg.norm(embeddings) / 10.0 
    
    # ΦC (Cohérence) : Synchronisation locale (stabilité spectrale simulée)
    phi_c = np.mean(np.abs(np.fft.fft(embeddings))) / 5.0
    
    # ΦD (Dissipation) : Évacuation de l'entropie (gradient inverse)
    phi_d = 1.0 - (np.std(embeddings))
    
    norm = math.sqrt(phi_m**2 + phi_c**2 + phi_d**2) + 1e-9
    return phi_m/norm, phi_c/norm, phi_d/norm

def triadic_entropy(text, beta=0.5):
    [span_6](start_span)[span_7](start_span)"""Calcule HΦ(X) intégrant la profondeur historique (Mémoire)[span_6](end_span)[span_7](end_span)"""
    phi_m, _, _ = get_triadic_metrics(text)
    # Terme 1: Incertitude classique (Shannon simplifiée sur caractères)
    counts = Counter(text)
    probs = [c/len(text) for c in counts.values()]
    shannon = -sum(p * math.log2(p) for p in probs)
    
    # Terme 2: Profondeur historique pondérée par beta
    h_phi = shannon - beta * math.log(max(phi_m, 1e-5))
    return h_phi

# =========================================================
# ÉCHELLES DE LYAPUNOV & STABILISATION
# =========================================================
def get_lyapunov_scale(weight):
    [span_8](start_span)[span_9](start_span)"""Classe les trajectoires selon les exposants λi[span_8](end_span)[span_9](end_span)"""
    if weight > 2.0: return "Stable (λ < 0)" # Logique & Strict
    if weight > 1.0: return "Neutre (λ ≈ 0)" # Cohérente
    return "Exploratrice (λ > 0)"            # Perspectives

def check_stability(m, c, d):
    [span_10](start_span)[span_11](start_span)"""Vérifie l'inégalité de stabilité : λM*λC*λD > κ[span_10](end_span)[span_11](end_span)"""
    # κ est ici simplifié à une constante de couplage critique
    return (m * c * d) > 0.01

# =========================================================
# INTERFACE STREAMLIT V14
# =========================================================
st.title("🧬 ORACLE Ω-TTU V14 : Systèmes Triadiques Multilingues")

if "phi" not in st.session_state:
    st.session_state.phi = {"m": 0.5, "c": 0.5, "d": 0.5}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Encodage & Apprentissage")
    input_text = st.text_area("Entrez des données (Texte, Hiéroglyphes, Nzebi, etc.)", height=150)
    
    if st.button("Transformer en Champ Informationnel"):
        if input_text:
            m, c, d = get_triadic_metrics(input_text)
            h_phi = triadic_entropy(input_text)
            
            st.session_state.phi = {"m": m, "c": c, "d": d}
            
            st.metric("Entropie Triadique HΦ", round(h_phi, 4))
            st.info(f"État du système : M={round(m,2)}, C={round(c,2)}, D={round(d,2)}")
            
            if check_stability(m, c, d):
                st.success("✅ Système stable : Prêt pour une réponse logique.")
            else:
                st.warning("⚠️ Dissipation insuffisante : Risque de bruit informationnel.")

with col2:
    st.subheader("📊 Analyse des Échelles de Lyapunov")
    # Simulation de trajectoires basées sur les poids de la DB
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT next_word, weight FROM ngrams LIMIT 10", conn)
        if not df.empty:
            df['Échelle'] = df['weight'].apply(get_lyapunov_scale)
            st.table(df)
        else:
            st.write("Aucune donnée dans la mémoire. Nourrissez l'Oracle.")
        conn.close()

# Visualisation du Cube de Cohérence
st.subheader("🧊 Espace de Hilbert Triadique (TTU-MC³)")
m, c, d = st.session_state.phi.values()
chart_data = pd.DataFrame({"Composante": ["Mémoire", "Cohérence", "Dissipation"], "Intensité": [m, c, d]})
st.bar_chart(chart_data, x="Composante", y="Intensité")

[span_12](start_span)[span_13](start_span)st.caption("Version V14 : Implémentation du Théorème de Convergence Topologique (TCT)[span_12](end_span)[span_13](end_span)")
