import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time, base64, math
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

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="ORACLE Ω-TTU V11.2", layout="wide", page_icon="🧠")

MEM_DIR = "oracle_memory"
os.makedirs(MEM_DIR, exist_ok=True)

FILES = {
    "fragments": f"{MEM_DIR}/fragments.csv",
    "concepts": f"{MEM_DIR}/concepts.csv",
    "relations": f"{MEM_DIR}/relations.json",
    "intentions": f"{MEM_DIR}/intentions.csv",
    "cortex": f"{MEM_DIR}/cortex.json"
}

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
BRANCH = "main"

# =========================================================
# INITIALISATION DES FICHIERS
# =========================================================
def init_files():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment", "count"]).to_csv(FILES["fragments"], index=False)
    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept", "weight"]).to_csv(FILES["concepts"], index=False)
    if not os.path.exists(FILES["intentions"]):
        pd.DataFrame(columns=["intent", "count"]).to_csv(FILES["intentions"], index=False)
    if not os.path.exists(FILES["relations"]):
        json.dump({}, open(FILES["relations"], "w"))
    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "VS": 12,
            "age": 0,
            "new_today": 0,
            "last_day": str(datetime.date.today()),
            "timeline": []
        }, open(FILES["cortex"], "w"))

init_files()

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
        st.session_state.shadow_rel = load_json(FILES["relations"])
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
        with open(FILES["relations"], "rb") as f:
            content = base64.b64encode(f.read()).decode()
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILES['relations']}"
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
    M = st.session_state.shadow_rel
    new = {}
    for w, con in M.items():
        filt = {t: v * 0.997 for t, v in con.items() if v > 1.2}
        if filt:
            new[w] = filt
    st.session_state.shadow_rel = new
    save_json(FILES["relations"], new)
    st.session_state.last_sleep = time.time()

# =========================================================
# TRAITEMENT DU TEXTE
# =========================================================
def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

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
# APPRENTISSAGE (HIPPOCAMPE)
# =========================================================
def learn(text):
    words = tokenize(text)
    if len(words) < 2:
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

    # Mise à jour associations (bigrammes)
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + (1.0 + st.session_state.phi["phi_m"])

    # Cortex et timeline
    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex.get("last_day") != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today
    cortex["age"] = cortex.get("age", 0) + len(words)
    cortex["new_today"] = cortex.get("new_today", 0) + len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"]))
    cortex.setdefault("timeline", []).extend(words)

    save_json(FILES["relations"], assoc)
    save_json(FILES["cortex"], cortex)
    st.session_state.shadow_rel = assoc
    st.session_state.shadow_cortex = cortex

    # Hippocampe pour consolidation différée
    energy = math.sqrt(sum(v*v for v in st.session_state.phi.values()))
    st.session_state.hippocampus.append((words, energy))
    if len(st.session_state.hippocampus) > 5 and consolidation_gate():
        consolidate()

    return len(words)

def consolidate():
    """Consolidation depuis l'hippocampe vers la mémoire long terme"""
    assoc = st.session_state.shadow_rel
    for words, energy in st.session_state.hippocampus:
        for a, b in zip(words, words[1:]):
            assoc.setdefault(a, {})
            assoc[a][b] = assoc[a].get(b, 0) + energy
    save_json(FILES["relations"], assoc)
    st.session_state.shadow_rel = assoc
    st.session_state.hippocampus.clear()
    github_sync()

# =========================================================
# GÉNÉRATION DE PENSÉE
# =========================================================
def contextual_seed():
    ctx = " ".join(st.session_state.dialog).split()
    valid = [w for w in ctx if w in st.session_state.shadow_rel]
    if valid:
        return Counter(valid).most_common(1)[0][0]
    all_words = list(st.session_state.shadow_rel.keys())
    return random.choice(all_words) if all_words else ""

def associative_layer(word):
    if word not in st.session_state.shadow_rel:
        return word
    opts = st.session_state.shadow_rel[word]
    if random.random() < st.session_state.phi["phi_c"]:
        # Choix probabiliste (bifurcation)
        return random.choices(list(opts.keys()), weights=list(opts.values()))[0]
    else:
        # Chemin le plus probable (cohérence)
        return max(opts, key=opts.get)

def think():
    if not st.session_state.shadow_rel:
        return "Mémoire vide. Nourrissez-moi de textes."
    seed = contextual_seed()
    if not seed:
        return "Je ne trouve pas de point de départ."
    words = [seed]
    length = int(10 + st.session_state.phi["phi_m"] * 30)
    for _ in range(length):
        nxt = associative_layer(words[-1])
        if nxt == words[-1]:  # évite boucle infinie
            break
        words.append(nxt)
    return " ".join(words).capitalize() + "."

# =========================================================
# ANALYSE SPECTRALE (si disponible)
# =========================================================
def build_signal_from_timeline(word):
    cortex = st.session_state.shadow_cortex
    timeline = cortex.get("timeline", [])
    return np.array([1 if w == word else 0 for w in timeline])

def spectral_analysis(word, nperseg=256):
    if not SPECTRAL_AVAILABLE:
        st.error("Les bibliothèques scipy et/ou matplotlib ne sont pas installées.")
        return None
    signal = build_signal_from_timeline(word)
    if len(signal) < nperseg:
        st.warning(f"Signal trop court ({len(signal)}). Besoin d'au moins {nperseg} mots.")
        return None
    fs = 1.0
    f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg//2)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud')
    ax1.set_ylabel('Fréquence [cycles/mot]')
    ax1.set_xlabel('Temps [mot]')
    ax1.set_title(f'Spectrogramme du mot "{word}"')

    # Fréquence dominante
    mean_amp = np.mean(np.abs(Zxx), axis=1)
    idx_max = np.argmax(mean_amp[1:]) + 1
    freq_dominant = f[idx_max]
    phase = np.angle(Zxx[idx_max, :])
    phase_unwrapped = np.unwrap(phase)

    # Amortissement α
    peak_amp = mean_amp[idx_max]
    half_amp = peak_amp / np.sqrt(2)
    left = np.where(mean_amp[:idx_max] <= half_amp)[0]
    right = np.where(mean_amp[idx_max:] <= half_amp)[0]
    if len(left) > 0 and len(right) > 0:
        f_left = f[left[-1]]
        f_right = f[idx_max + right[0]]
        bandwidth = f_right - f_left
        alpha = bandwidth / 2
    else:
        alpha = 0.0

    # Phase
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, phase_unwrapped, 'b-')
    ax2.set_xlabel('Temps [mot]')
    ax2.set_ylabel('Phase déroulée [rad]')
    ax2.set_title(f'Phase à f = {freq_dominant:.4f} cycles/mot')

    # Linéarité de phase
    if len(t) > 1:
        coeffs = np.polyfit(t, phase_unwrapped, 1)
        trend = np.polyval(coeffs, t)
        residuals = phase_unwrapped - trend
        linearity = 1 - np.std(residuals) / (np.std(phase_unwrapped) + 1e-10)
    else:
        linearity = 0.0

    results = {
        "omega": 2 * np.pi * freq_dominant,
        "alpha": alpha,
        "lambda": complex(-alpha, 2*np.pi*freq_dominant),
        "freq_dominant": freq_dominant,
        "linearity": linearity,
        "signal_length": len(signal),
        "nperseg": nperseg
    }
    return {"results": results, "figures": (fig1, fig2)}

# =========================================================
# AUTO-DIAGNOSTIC
# =========================================================
def diagnose():
    cortex = st.session_state.shadow_cortex
    assoc = st.session_state.shadow_rel
    links = sum(len(v) for v in assoc.values())
    vocab = len(assoc)
    density = round(links / max(vocab, 1), 2)

    if cortex.get("new_today", 0) < 20:
        return "🧠 J'ai besoin de nouvelles connaissances."
    if density < 1.5:
        return "🧠 Donne-moi des textes plus longs."
    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."
    return "🧠 Apprentissage actif."

# =========================================================
# TÉLÉCHARGEMENT DES DONNÉES
# =========================================================
def get_download_data():
    return {
        "fragments": st.session_state.shadow_frag.to_csv(index=False),
        "concepts": st.session_state.shadow_concepts.to_csv(index=False),
        "relations": json.dumps(st.session_state.shadow_rel, indent=2, ensure_ascii=False),
        "cortex": json.dumps(st.session_state.shadow_cortex, indent=2, ensure_ascii=False)
    }

# =========================================================
# INTERFACE UTILISATEUR
# =========================================================
st.title("🧠 ORACLE Ω-TTU V11.2 — Agent Cognitif Spectral")
st.caption("Projection Triadique + Mémoire TST + Analyse Spectrale Temps-Fréquence")

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
    if st.button("⬇️ Télécharger données"):
        data = get_download_data()
        st.download_button("fragments.csv", data["fragments"], "fragments.csv")
        st.download_button("concepts.csv", data["concepts"], "concepts.csv")
        st.download_button("relations.json", data["relations"], "relations.json")
        st.download_button("cortex.json", data["cortex"], "cortex.json")

    st.divider()
    st.info(diagnose())

# Onglets pour séparer l'apprentissage et la conversation
tab1, tab2, tab3 = st.tabs(["🌱 Nourrir", "💬 Parler", "📊 Analyse Spectrale"])

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
            learn(content)
            st.success(f"Apprentissage effectué ({len(tokenize(content))} mots).")
            st.rerun()
        else:
            st.warning("Aucun contenu à apprendre.")

with tab2:
    st.subheader("Conversation")
    user_msg = st.text_input("Votre message")
    if st.button("Envoyer") and user_msg:
        st.session_state.dialog.append("👤 " + user_msg)
        excitation = min(1.0, len(user_msg) / 200)
        st.session_state.phi = evolve_ttu(st.session_state.phi, excitation)
        learn(user_msg)
        reply = think()
        st.session_state.dialog.append("🧠 " + reply)
        st.rerun()

    # Affichage de la conversation
    for msg in st.session_state.dialog:
        st.write(msg)

with tab3:
    st.subheader("Analyse Spectrale d'un Concept")
    if not SPECTRAL_AVAILABLE:
        st.error("Analyse spectrale désactivée : installer scipy et matplotlib.")
    else:
        words = list(st.session_state.shadow_rel.keys())
        if words:
            word = st.selectbox("Choisir un mot", words)
            nperseg = st.slider("Taille de fenêtre STFT", 64, 512, 256, 32)
            if st.button("Lancer l'analyse"):
                with st.spinner("Calcul en cours..."):
                    result = spectral_analysis(word, nperseg)
                if result:
                    st.json(result["results"])
                    fig1, fig2 = result["figures"]
                    st.pyplot(fig1)
                    st.pyplot(fig2)
        else:
            st.info("Aucun mot en mémoire pour l'analyse.")

# =========================================================
# PIED DE PAGE (AFFICHAGE DE LA TAILLE DE LA MÉMOIRE)
# =========================================================
st.divider()
mem_size = os.path.getsize(FILES["relations"]) / 1024
st.caption(f"Mémoire : {mem_size:.2f} Ko | {len(st.session_state.shadow_rel)} concepts")