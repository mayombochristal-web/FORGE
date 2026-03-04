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
    # Modèle léger pour embeddings de mots (on peut aussi utiliser FastText)
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
st.set_page_config(page_title="ORACLE Ω-TTU V12", layout="wide", page_icon="🧠")

MEM_DIR = "oracle_memory"
os.makedirs(MEM_DIR, exist_ok=True)

# Utilisation de SQLite pour les relations (trigrammes)
DB_PATH = os.path.join(MEM_DIR, "relations.db")

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
# INITIALISATION DE LA BASE SQLITE
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table des trigrammes : (w1, w2) -> w3 avec poids
    c.execute('''
        CREATE TABLE IF NOT EXISTS trigrams (
            w1 TEXT,
            w2 TEXT,
            w3 TEXT,
            weight REAL,
            PRIMARY KEY (w1, w2, w3)
        )
    ''')
    # Index pour accélérer les requêtes
    c.execute('CREATE INDEX IF NOT EXISTS idx_w1_w2 ON trigrams (w1, w2)')
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
        # On ne charge pas toutes les relations en mémoire, on utilisera des requêtes SQL
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
# SYNCHRONISATION GITHUB (optionnelle)
# =========================================================
def github_sync():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        # On sauvegarde un dump JSON de la base SQLite (version simplifiée)
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trigrams", conn)
        conn.close()
        data = df.to_dict(orient='records')
        content = base64.b64encode(json.dumps(data).encode()).decode()
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_DIR}/relations_dump.json"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=10)
        sha = r.json()["sha"] if r.status_code == 200 else None
        data = {
            "message": "🧬 Oracle memory auto-sync",
            "content": content,
            "branch": BRANCH
        }
        if sha:
            data["sha"] = sha
        requests.put(url, headers=headers, json=data, timeout=10)
    except Exception as e:
        st.warning(f"Sync GitHub échoué : {e}")

# =========================================================
# CYCLE DE SOMMEIL (CONSOLIDATION)
# =========================================================
def sleep_cycle():
    # Nettoyage entropique : on supprime les trigrammes de poids faible
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM trigrams WHERE weight < 1.2")
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
# CALCUL DU IDF (pour pondération TF-IDF)
# =========================================================
def compute_idf():
    df = st.session_state.shadow_frag
    total_docs = len(df) if len(df) > 0 else 1
    word_doc_count = {}
    # Approximation : chaque fragment est un document (chaque ligne de fragments.csv)
    for idx, row in df.iterrows():
        words = row['fragment'].split()  # mais ce n'est pas un document complet
    # On va plutôt utiliser le cortex timeline comme ensemble de documents ?
    # Simplification : on utilise les bigrammes/trigrammes présents dans la base.
    # Pour l'instant, on renvoie un dictionnaire vide (pas de TF-IDF).
    return {}

# =========================================================
# OBTENIR L'EMBEDDING D'UN MOT (si disponible)
# =========================================================
def get_word_embedding(word):
    if EMBEDDINGS_AVAILABLE:
        return embed_model.encode(word)
    else:
        return None

# =========================================================
# APPRENTISSAGE (HIPPOCAMPE) avec trigrammes et embeddings
# =========================================================
def learn(text):
    words = tokenize(text)
    if len(words) < 3:  # besoin d'au moins 3 mots pour trigramme
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

    # Mise à jour des trigrammes dans SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # On parcourt les séquences de 3 mots
    for i in range(len(words)-2):
        a, b, d = words[i], words[i+1], words[i+2]
        # Incrément du poids (avec Φm)
        inc = 1.0 + st.session_state.phi["phi_m"]
        # Vérifier si le triplet existe déjà
        c.execute('''
            INSERT INTO trigrams (w1, w2, w3, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(w1, w2, w3) DO UPDATE SET weight = weight + excluded.weight
        ''', (a, b, d, inc))
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

    # Hippocampe pour consolidation différée (on garde les mots pour renforcement)
    energy = math.sqrt(sum(v*v for v in st.session_state.phi.values()))
    st.session_state.hippocampus.append((words, energy))
    if len(st.session_state.hippocampus) > 5 and consolidation_gate():
        consolidate()

    return len(words)

def consolidate():
    """Consolidation depuis l'hippocampe (actuellement inutilisé pour les trigrammes, mais on pourrait renforcer)"""
    # Ici on pourrait renforcer les trigrammes déjà présents
    st.session_state.hippocampus.clear()
    github_sync()

# =========================================================
# RECHERCHE DES SUIVANTS POSSIBLES (TRIGRAMMES)
# =========================================================
def get_next_candidates(w1, w2):
    """Retourne un dictionnaire {mot3: poids} pour le bigramme (w1, w2)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT w3, weight FROM trigrams WHERE w1=? AND w2=?", (w1, w2))
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

# =========================================================
# FONCTIONS D'ANALYSE SPECTRALE (pour l'attention)
# =========================================================
def get_spectral_features(word):
    """
    Calcule la fréquence dominante et la stabilité de phase pour un mot donné à partir de la timeline.
    Retourne (freq_dom, phase_stability) où phase_stability est la linéarité de la phase (0-1).
    """
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
    # Phase à la fréquence dominante
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
# GÉNÉRATION DE PENSÉE AVEC TEMPÉRATURE DYNAMIQUE ET ATTENTION SPECTRALE
# =========================================================
def contextual_seed():
    ctx = " ".join(st.session_state.dialog).split()
    # On cherche un bigramme connu dans la base
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for i in range(len(ctx)-1):
        a, b = ctx[i], ctx[i+1]
        c.execute("SELECT 1 FROM trigrams WHERE w1=? AND w2=? LIMIT 1", (a, b))
        if c.fetchone():
            conn.close()
            return a, b
    conn.close()
    # Sinon on prend un bigramme aléatoire dans la base
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT w1, w2 FROM trigrams ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def think():
    # Récupérer le contexte pour le seed
    w1, w2 = contextual_seed()
    if w1 is None:
        return "Mémoire vide. Nourrissez-moi de textes."

    words = [w1, w2]
    length = int(10 + st.session_state.phi["phi_m"] * 30)

    for _ in range(length):
        candidates = get_next_candidates(words[-2], words[-1])
        if not candidates:
            break

        # Calcul de la température dynamique liée à Φd
        temp = max(0.5, 1.5 * st.session_state.phi["phi_d"])

        # Calcul des scores bruts (poids)
        items = list(candidates.items())
        words_list = [w for w, _ in items]
        weights = np.array([w for _, w in items], dtype=float)

        # Application de l'attention spectrale si disponible
        if SPECTRAL_AVAILABLE:
            for i, w in enumerate(words_list):
                freq, stab = get_spectral_features(w)
                # On favorise les mots dont la fréquence dominante est proche de celle du contexte ? (simplification)
                # Ici on utilise la stabilité comme boost (plus c'est stable, plus on le favorise)
                weights[i] *= (0.5 + stab)  # stab entre 0 et 1

        # Normalisation avec température
        weights = weights ** (1.0 / temp)
        probs = weights / weights.sum()

        # Choix du prochain mot
        next_word = np.random.choice(words_list, p=probs)
        words.append(next_word)

        # Mise à jour du bigramme courant
        w1, w2 = words[-2], words[-1]

    return " ".join(words).capitalize() + "."

# =========================================================
# FEEDBACK UTILISATEUR (ACTIVE LEARNING)
# =========================================================
def apply_feedback(reply, is_positive):
    """
    reply est la phrase générée (liste de mots). On renforce ou affaiblit les trigrammes utilisés.
    """
    words = tokenize(reply)
    if len(words) < 3:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    delta = 0.5 if is_positive else -0.3
    for i in range(len(words)-2):
        a, b, d = words[i], words[i+1], words[i+2]
        c.execute('''
            UPDATE trigrams SET weight = weight + ? WHERE w1=? AND w2=? AND w3=?
        ''', (delta, a, b, d))
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
    c.execute("SELECT COUNT(*) FROM trigrams")
    nb_trigrams = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT w1 || w2) FROM trigrams")
    nb_bigrams = c.fetchone()[0]
    conn.close()
    density = round(nb_trigrams / max(nb_bigrams, 1), 2)

    if cortex.get("new_today", 0) < 20:
        return "🧠 J'ai besoin de nouvelles connaissances."
    if density < 1.5:
        return "🧠 Donne-moi des textes plus longs."
    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."
    return "🧠 Apprentissage actif."

# =========================================================
# VISUALISATION 2D DES CONCEPTS (T-SNE / PCA)
# =========================================================
def plot_concepts_2d():
    if not VIZ_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        st.warning("Visualisation nécessite scikit-learn, plotly et sentence-transformers.")
        return

    # Récupérer tous les mots distincts de la base
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT w1 FROM trigrams UNION SELECT DISTINCT w2 FROM trigrams UNION SELECT DISTINCT w3 FROM trigrams")
    words = [row[0] for row in c.fetchall()]
    conn.close()

    if len(words) < 5:
        st.info("Pas assez de concepts pour la visualisation.")
        return

    # Calculer les embeddings
    embeddings = embed_model.encode(words)

    # Réduction dimensionnelle (PCA d'abord, puis TSNE pour la visualisation)
    pca = PCA(n_components=min(50, len(embeddings)))
    pca_result = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=min(30, len(words)-1))
    tsne_result = tsne.fit_transform(pca_result)

    df = pd.DataFrame({'word': words, 'x': tsne_result[:,0], 'y': tsne_result[:,1]})
    fig = px.scatter(df, x='x', y='y', text='word', title='Carte sémantique 2D des concepts')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# INTERFACE UTILISATEUR
# =========================================================
st.title("🧠 ORACLE Ω-TTU V12 — Agent Cognitif Spectral")
st.caption("Trigrammes + Embeddings + Température dynamique + Feedback actif")

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
    if st.button("☁️ Sync GitHub"):
        github_sync()
        st.success("Mémoire synchronisée.")
    if st.button("⬇️ Télécharger données (JSON)"):
        # Export de la base SQLite en CSV
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trigrams", conn)
        conn.close()
        st.download_button("trigrams.csv", df.to_csv(index=False), "trigrams.csv")

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
            st.success(f"Apprentissage effectué ({nb_words} mots).")
            st.rerun()
        else:
            st.warning("Aucun contenu à apprendre.")

with tab2:
    st.subheader("Conversation")

    # Zone d'affichage des messages
    for msg in st.session_state.dialog:
        st.write(msg)

    user_msg = st.text_input("Votre message", key="user_input")
    col1, col2 = st.columns([1,5])
    with col1:
        send = st.button("Envoyer")
    if send and user_msg:
        st.session_state.dialog.append("👤 " + user_msg)
        excitation = min(1.0, len(user_msg) / 200)
        st.session_state.phi = evolve_ttu(st.session_state.phi, excitation)
        learn(user_msg)
        reply = think()
        st.session_state.dialog.append("🧠 " + reply)

        # Stocker la dernière réponse pour le feedback
        st.session_state.last_reply = reply
        st.session_state.last_reply_words = tokenize(reply)

        st.rerun()

    # Boutons de feedback (apparaissent après une réponse)
    if "last_reply" in st.session_state:
        st.markdown("**Cette réponse était-elle pertinente ?**")
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("👍 Pertinent"):
                apply_feedback(st.session_state.last_reply, True)
                st.success("Feedback enregistré (renforcement).")
                del st.session_state.last_reply
                st.rerun()
        with fb2:
            if st.button("👎 Non pertinent"):
                apply_feedback(st.session_state.last_reply, False)
                st.warning("Feedback enregistré (affaiblissement).")
                del st.session_state.last_reply
                st.rerun()

with tab3:
    st.subheader("Analyse Spectrale d'un Concept")
    if not SPECTRAL_AVAILABLE:
        st.error("Analyse spectrale désactivée : installer scipy et matplotlib.")
    else:
        # Récupérer la liste des mots distincts
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT DISTINCT w1 FROM trigrams UNION SELECT DISTINCT w2 FROM trigrams UNION SELECT DISTINCT w3 FROM trigrams")
        words = [row[0] for row in c.fetchall()]
        conn.close()
        if words:
            word = st.selectbox("Choisir un mot", words)
            nperseg = st.slider("Taille de fenêtre STFT", 64, 512, 256, 32)
            if st.button("Lancer l'analyse"):
                with st.spinner("Calcul en cours..."):
                    signal = np.array([1 if w == word else 0 for w in st.session_state.shadow_cortex.get("timeline", [])])
                    if len(signal) < nperseg:
                        st.warning(f"Signal trop court ({len(signal)}). Besoin d'au moins {nperseg} mots.")
                    else:
                        fs = 1.0
                        f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg//2)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud')
                        ax.set_ylabel('Fréquence [cycles/mot]')
                        ax.set_xlabel('Temps [mot]')
                        ax.set_title(f'Spectrogramme du mot "{word}"')
                        st.pyplot(fig)
        else:
            st.info("Aucun mot en mémoire pour l'analyse.")

with tab4:
    st.subheader("Neuro-imagerie : Carte sémantique 2D")
    if st.button("Générer la carte"):
        with st.spinner("Calcul des embeddings et réduction dimensionnelle..."):
            plot_concepts_2d()

# =========================================================
# PIED DE PAGE
# =========================================================
st.divider()
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM trigrams")
nb_tri = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT w1) FROM trigrams")
nb_vocab = c.fetchone()[0]
conn.close()
st.caption(f"Mémoire : {nb_tri} trigrammes | {nb_vocab} mots distincts | Φ = {st.session_state.phi}")