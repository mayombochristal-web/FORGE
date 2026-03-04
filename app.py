import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time, base64, math, sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, deque
import random
import requests

# =========================================================
# GESTION DES DÉPENDANCES OPTIONNELLES
# =========================================================
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    from scipy.signal import stft
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import plotly.express as px
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="ORACLE Ω-TTU V13", layout="wide", page_icon="🧠")

MEM_DIR = "oracle_memory"
os.makedirs(MEM_DIR, exist_ok=True)

DB_PATH = os.path.join(MEM_DIR, "relations.db")
MAX_N = 5  # Longueur maximale des n-grammes (contexte de taille MAX_N-1)

FILES = {
    "fragments": f"{MEM_DIR}/fragments.csv",
    "concepts": f"{MEM_DIR}/concepts.csv",
    "intentions": f"{MEM_DIR}/intentions.csv",
    "cortex": f"{MEM_DIR}/cortex.json"
}

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
BRANCH = "main"

# =========================================================
# INITIALISATION DE LA BASE SQLITE (n-grammes)
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Supprimer l'ancienne table trigrams si elle existe
    c.execute("DROP TABLE IF EXISTS trigrams")
    # Créer la nouvelle table ngrams
    c.execute('''
        CREATE TABLE IF NOT EXISTS ngrams (
            context TEXT,
            next_word TEXT,
            weight REAL,
            PRIMARY KEY (context, next_word)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_context ON ngrams (context)')
    conn.commit()
    conn.close()

init_db()

# =========================================================
# CHARGEMENT DES DONNÉES EN SESSION (SHADOW STATE)
# =========================================================
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p, d):
    with open(p, "w") as f:
        json.dump(d, f)

def load_frag():
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.reset_index(drop=True).to_csv(FILES["fragments"], index=False)

def load_concepts():
    return pd.read_csv(FILES["concepts"])

def save_concepts(df):
    df.reset_index(drop=True).to_csv(FILES["concepts"], index=False)

def sync_shadow():
    if "shadow_loaded" not in st.session_state:
        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_concepts = load_concepts().copy()
        st.session_state.shadow_cortex = load_json(FILES["cortex"])
        st.session_state.shadow_loaded = True

sync_shadow()

# =========================================================
# ÉTAT COGNITIF (Φ TRIADIQUE)
# =========================================================
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

if "dialog" not in st.session_state:
    st.session_state.dialog = deque(maxlen=60)

if "hippocampus" not in st.session_state:
    st.session_state.hippocampus = []

if "green_state" not in st.session_state:
    st.session_state.green_state = 0.0

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep = time.time()

# =========================================================
# FONCTIONS TTU (PROJECTION TRIADIQUE)
# =========================================================
def normalize_ttu(m, c, d):
    norm = math.sqrt(m**2 + c**2 + d**2) + 1e-9
    return m/norm, c/norm, d/norm

def evolve_ttu(phi, excitation):
    new_m = min(1.0, max(0.1, phi["phi_m"] + excitation * 0.12 - 0.02))
    new_c = min(1.0, max(0.1, phi["phi_c"] + excitation * 0.25 - 0.04))
    new_d = min(1.0, max(0.1, phi["phi_d"] + 0.05 - excitation * 0.08))
    m, c, d = normalize_ttu(new_m, new_c, new_d)
    return {"phi_m": m, "phi_c": c, "phi_d": d}

# =========================================================
# GREEN NOISE (HOMÉOSTASIE)
# =========================================================
def green_noise(prev):
    return 0.92 * prev + 0.08 * random.uniform(-1, 1)

def consolidation_gate():
    st.session_state.green_state = green_noise(st.session_state.green_state)
    return abs(st.session_state.green_state) < 0.25

# =========================================================
# SYNCHRONISATION GITHUB
# =========================================================
def github_sync():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM ngrams", conn)
        conn.close()
        data = df.to_dict(orient='records')
        content = base64.b64encode(json.dumps(data).encode()).decode()
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_DIR}/relations_dump.json"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=10)
        sha = r.json()["sha"] if r.status_code == 200 else None
        payload = {
            "message": "🧬 Oracle memory auto-sync",
            "content": content,
            "branch": BRANCH
        }
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        st.warning(f"Sync GitHub échoué : {e}")

# =========================================================
# CYCLE DE SOMMEIL (CONSOLIDATION)
# =========================================================
def sleep_cycle():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM ngrams WHERE weight < 1.2")
    conn.commit()
    conn.close()
    st.session_state.last_sleep = time.time()

# =========================================================
# TRAITEMENT DU TEXTE
# =========================================================
def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

# =========================================================
# APPRENTISSAGE (n-grammes) avec importance
# =========================================================
def learn(text, importance=1.0):
    words = tokenize(text)
    if len(words) < 2:  # besoin d'au moins 2 mots pour un bigramme
        return 0

    # Mise à jour fragments
    df = st.session_state.shadow_frag.copy()
    counts = Counter(words)
    for w, c in counts.items():
        mask = df["fragment"] == w
        if mask.any():
            df.loc[mask, "count"] += c
        else:
            df = pd.concat([df, pd.DataFrame([[w, c]], columns=df.columns)], ignore_index=True)
    save_frag(df)
    st.session_state.shadow_frag = df

    # Mise à jour des n-grammes dans SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for n in range(2, MAX_N+1):  # de 2 à MAX_N
        for i in range(len(words)-n+1):
            context_words = words[i:i+n-1]
            next_word = words[i+n-1]
            context = " ".join(context_words)
            inc = importance * (1.0 + st.session_state.phi["phi_m"])
            c.execute('''
                INSERT INTO ngrams (context, next_word, weight)
                VALUES (?, ?, ?)
                ON CONFLICT(context, next_word) DO UPDATE SET weight = weight + excluded.weight
            ''', (context, next_word, inc))
    conn.commit()
    conn.close()

    # Mise à jour du cortex et timeline
    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex.get("last_day") != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today
    cortex["age"] = cortex.get("age", 0) + len(words)
    cortex["new_today"] = cortex.get("new_today", 0) + len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"]))
    cortex.setdefault("timeline", []).extend(words)

    save_json(FILES["cortex"], cortex)
    st.session_state.shadow_cortex = cortex

    energy = math.sqrt(sum(v*v for v in st.session_state.phi.values()))
    st.session_state.hippocampus.append((words, energy))
    if len(st.session_state.hippocampus) > 5 and consolidation_gate():
        consolidate()

    return len(words)

def consolidate():
    st.session_state.hippocampus.clear()
    github_sync()

# =========================================================
# RECHERCHE DES SUIVANTS POSSIBLES POUR UN CONTEXTE DONNÉ
# =========================================================
def get_candidates(context):
    """Retourne un dictionnaire {mot: poids} pour un contexte exact"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT next_word, weight FROM ngrams WHERE context=?", (context,))
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

# =========================================================
# FONCTIONS D'ANALYSE SPECTRALE (pour l'attention)
# =========================================================
def get_spectral_features(word):
    if not SPECTRAL_AVAILABLE:
        return 0.0, 0.5
    timeline = st.session_state.shadow_cortex.get("timeline", [])
    if len(timeline) < 256:
        return 0.0, 0.5
    signal = np.array([1 if w == word else 0 for w in timeline])
    fs = 1.0
    f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=256, noverlap=128)
    mean_amp = np.mean(np.abs(Zxx), axis=1)
    idx_max = np.argmax(mean_amp[1:]) + 1
    freq_dom = f[idx_max]
    phase = np.angle(Zxx[idx_max, :])
    phase_unwrapped = np.unwrap(phase)
    if len(t) > 1:
        coeffs = np.polyfit(t, phase_unwrapped, 1)
        trend = np.polyval(coeffs, t)
        residuals = phase_unwrapped - trend
        stability = 1 - np.std(residuals) / (np.std(phase_unwrapped) + 1e-10)
    else:
        stability = 0.5
    return freq_dom, stability

# =========================================================
# GÉNÉRATION DE PENSÉE AVEC BACKOFF
# =========================================================
def get_best_context(words_list, max_len=MAX_N-1):
    """Trouve le plus long contexte existant dans la base (backoff)"""
    for l in range(min(max_len, len(words_list)), 0, -1):
        context = " ".join(words_list[-l:])
        cand = get_candidates(context)
        if cand:
            return context, cand
    return None, {}

def contextual_seed():
    # Cherche un bigramme existant dans la conversation récente
    ctx = " ".join(st.session_state.dialog).split()
    for i in range(len(ctx)-1, 0, -1):
        context = " ".join(ctx[max(0, i-1):i+1])  # bigramme
        if get_candidates(context):
            return context.split()  # retourne la liste des mots du contexte
    # Sinon, prend un contexte aléatoire dans la base
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT context FROM ngrams ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return row[0].split()
    return None

def think():
    seed_words = contextual_seed()
    if not seed_words:
        return "Mémoire vide. Nourrissez-moi de textes."

    words = seed_words.copy()
    length = int(10 + st.session_state.phi["phi_m"] * 30)

    for _ in range(length):
        context, candidates = get_best_context(words)
        if not candidates:
            break

        temp = max(0.5, 1.5 * st.session_state.phi["phi_d"])
        items = list(candidates.items())
        words_list = [w for w, _ in items]
        weights = np.array([w for _, w in items], dtype=float)

        if SPECTRAL_AVAILABLE:
            for i, w in enumerate(words_list):
                freq, stab = get_spectral_features(w)
                weights[i] *= (0.5 + stab)

        weights = weights ** (1.0 / temp)
        probs = weights / weights.sum()
        next_word = np.random.choice(words_list, p=probs)
        words.append(next_word)

    return " ".join(words).capitalize() + "."

# =========================================================
# FEEDBACK UTILISATEUR (ACTIVE LEARNING)
# =========================================================
def apply_feedback(reply, is_positive):
    words = tokenize(reply)
    if len(words) < 2:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    delta = 0.5 if is_positive else -0.3
    for n in range(2, min(MAX_N, len(words)+1)):
        for i in range(len(words)-n+1):
            context_words = words[i:i+n-1]
            next_word = words[i+n-1]
            context = " ".join(context_words)
            c.execute('''
                UPDATE ngrams SET weight = weight + ? WHERE context=? AND next_word=?
            ''', (delta, context, next_word))
    conn.commit()
    conn.close()

# =========================================================
# LECTURE DES FICHIERS UPLOADÉS (MULTIMODAL)
# =========================================================
def read_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            return file.read().decode("utf-8", "ignore")
        if name.endswith(".csv"):
            return pd.read_csv(file).to_string()
        if name.endswith(".xlsx"):
            return pd.read_excel(file).to_string()
        if name.endswith(".docx") and DOCX_AVAILABLE:
            doc = docx.Document(io.BytesIO(file.read()))
            return " ".join(p.text for p in doc.paragraphs)
        if name.endswith(".pdf") and PDF_AVAILABLE:
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            return " ".join(p.extract_text() or "" for p in reader.pages)
        if name.endswith(".wav") and SR_AVAILABLE:
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio = r.record(source)
            return r.recognize_google(audio, language="fr-FR")
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
    return ""

# =========================================================
# AUTO-DIAGNOSTIC
# =========================================================
def diagnose():
    cortex = st.session_state.shadow_cortex
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM ngrams")
    nb_ngrams = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT context) FROM ngrams")
    nb_contexts = c.fetchone()[0]
    conn.close()
    density = round(nb_ngrams / max(nb_contexts, 1), 2)

    if cortex.get("new_today", 0) < 20:
        return "🧠 J'ai besoin de nouvelles connaissances."
    if density < 1.5:
        return "🧠 Donne-moi des textes plus longs."
    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."
    return "🧠 Apprentissage actif."

# =========================================================
# VISUALISATION 2D DES CONCEPTS
# =========================================================
def plot_concepts_2d():
    if not VIZ_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        st.warning("Visualisation nécessite scikit-learn, plotly et sentence-transformers.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT next_word FROM ngrams")
    words = [row[0] for row in c.fetchall()]
    conn.close()

    if len(words) < 5:
        st.info("Pas assez de concepts pour la visualisation.")
        return

    embeddings = embed_model.encode(words)
    pca = PCA(n_components=min(50, len(embeddings)))
    pca_result = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=min(30, len(words)-1))
    tsne_result = tsne.fit_transform(pca_result)

    df = pd.DataFrame({'word': words, 'x': tsne_result[:,0], 'y': tsne_result[:,1]})
    fig = px.scatter(df, x='x', y='y', text='word', title='Carte sémantique 2D des concepts')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FONCTION POUR OBTENIR LE NOMBRE DE N-GRAMMES
# =========================================================
def get_ngram_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM ngrams")
    count = c.fetchone()[0]
    conn.close()
    return count

# =========================================================
# INTERFACE UTILISATEUR
# =========================================================
st.title("🧠 ORACLE Ω-TTU V13 — n-grammes généralisés")
st.caption("Contexte variable (jusqu'à 5 mots) + Backoff + Φ dynamique")

# Barre latérale : état cognitif
with st.sidebar:
    st.header("🔬 État Cognitif")
    p = st.session_state.phi
    st.metric("Mémoire (Φm)", f"{p['phi_m']:.2f}")
    st.progress(p['phi_m'])
    st.metric("Cohérence (Φc)", f"{p['phi_c']:.2f}")
    st.progress(p['phi_c'])
    st.metric("Dissipation (Φd)", f"{p['phi_d']:.2f}")
    st.progress(p['phi_d'])

    st.divider()
    st.metric("Âge (mots)", st.session_state.shadow_cortex.get("age", 0))
    st.metric("Nouveaux aujourd'hui", st.session_state.shadow_cortex.get("new_today", 0))
    st.metric("VS", f"{st.session_state.shadow_cortex.get('VS', 12):.1f}")

    st.divider()
    if st.button("🌙 Cycle de Sommeil"):
        sleep_cycle()
        st.success("Consolidation et entropie réduite.")

    if st.button("📥 Exporter les n-grammes (CSV)"):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM ngrams", conn)
        conn.close()
        st.download_button("Télécharger ngrams.csv", df.to_csv(index=False), "ngrams.csv")

    if st.button("☁️ Sauvegarder sur GitHub"):
        github_sync()
        st.success("Mémoire synchronisée avec GitHub.")

    st.divider()
    st.info(diagnose())

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["🌱 Nourrir", "💬 Parler", "📊 Analyse Spectrale", "🧬 Neuro-imagerie"])

with tab1:
    st.subheader("Apprentissage Multimodal")
    mode = st.radio("Source", ["Texte", "Document (PDF/DOCX/TXT)", "Excel", "Audio (WAV)"], horizontal=True)
    content = ""
    if mode == "Texte":
        content = st.text_area("Entrez un texte")
    else:
        file_types = []
        if mode == "Document (PDF/DOCX/TXT)":
            file_types = ["pdf", "docx", "txt"]
        elif mode == "Excel":
            file_types = ["xlsx"]
        elif mode == "Audio (WAV)":
            file_types = ["wav"]
        uploaded = st.file_uploader("Charger fichier", type=file_types)
        if uploaded:
            content = read_file(uploaded)
            if content:
                st.success("Fichier lu avec succès.")
            else:
                st.warning("Échec de la lecture (format non supporté ou bibliothèque manquante).")

    if st.button("🌱 Nourrir l'Oracle"):
        if content:
            excitation = min(1.0, len(content) / 200)
            st.session_state.phi = evolve_ttu(st.session_state.phi, excitation)
            nb_words = learn(content)
            nb_ng = get_ngram_count()
            st.success(f"Apprentissage effectué ({nb_words} mots). Nombre de n-grammes maintenant : {nb_ng}.")
            st.rerun()
        else:
            st.warning("Aucun contenu à apprendre.")

with tab2:
    st.subheader("Conversation”