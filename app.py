import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time, base64, math
import xml.etree.ElementTree as ET
from collections import Counter, deque
import random
import requests
import PyPDF2
import docx
import speech_recognition as sr
from scipy.signal import stft
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="ORACLE Ω-TTU V11.0", layout="wide", page_icon="🧠")

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
# LECTURE DES FICHIERS UPLOADÉS
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
        if name.endswith(".docx"):
            doc = zipfile.ZipFile(io.BytesIO(file.read()))
            xml = doc.read("word/document.xml")
            tree = ET.fromstring(xml)
            return " ".join(t.text for t in tree.iter() if t.text)
        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            return " ".join(p.extract_text() or "" for p in reader.pages)
        if name.endswith(".wav"):
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio = r.record(source)
            return r.recognize_google(audio)
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
        # Choix probabiliste (bifurcation de Morse-Smale)
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
# ANALYSE SPECTRALE
# =========================================================
def build_signal_from_timeline(word):
    cortex = st.session_state.shadow_cortex
    timeline = cortex.get("timeline", [])
    return np.array([1 if w == word else 0 for w in timeline])

def spectral_analysis(word, nperseg=256):
    signal = build_signal_from_timeline(word)
    if len(signal) < nperseg:
        return {"error": f"Signal trop court ({len(signal)}). Besoin d'au moins {nperseg} mots."}
    fs = 1.0
    f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg//2)

    # Spectrogramme
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
    coherence = round(min(100, (links / max(len(st.session_state.shadow_concepts), 1)) * 10), 2)

    if cortex.get("new_today", 0) < 20:
        return "🧠 J'ai besoin de nouvelles connaissances."
    if density < 1.5:
        return "🧠 Donne-moi des textes plus longs."
    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."
    if coherence > 50:
        return "🧠 Cohérence sémantique élevée."
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
st.title("🧠 ORACLE Ω-TTU V11.0 — Agent Cognitif Spectral")
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

# Zone de dialogue
chat_container = st.container()
with chat_container:
    for msg in st.session_state.dialog:
        if msg.startswith("👤"):
            st.markdown(f"**{msg}**")
        else:
            st.markdown(f"🧠 *{msg}*")

# Saisie utilisateur
with st.container():
    st.divider()
    c1, c2 = st.columns([8, 1])
    user_input = c1.text_input("Parlez à l'Oracle", placeholder="Saisissez un texte ou posez une question...")
    send_btn = c2.button("Envoyer")

    uploaded_file = st.file_uploader("📥 Injecter document / audio", type=["txt", "pdf", "docx", "csv", "xlsx", "wav"])

# Traitement de l'entrée
if send_btn or user_input or uploaded_file:
    msg = ""
    if uploaded_file:
        msg = read_file(uploaded_file)
    elif user_input:
        msg = user_input

    if msg:
        # Excitation
        excitation = min(1.0, len(msg) / 200)
        st.session_state.phi = evolve_ttu(st.session_state.phi, excitation)

        # Apprentissage
        st.session_state.dialog.append("👤 " + msg[:200] + ("..." if len(msg) > 200 else ""))
        learn(msg)

        # Génération de réponse
        reply = think()
        st.session_state.dialog.append(reply)

        # Mise à jour shadow (déjà fait dans learn) et rafraîchissement
        st.rerun()

# Analyse spectrale (optionnel)
if st.checkbox("📊 Afficher l'analyse spectrale d'un concept"):
    st.subheader("Analyse Spectrale TST")
    words = list(st.session_state.shadow_rel.keys())
    if words:
        word = st.selectbox("Choisir un mot", words)
        nperseg = st.slider("Taille de fenêtre STFT", 64, 512, 256, 32)
        if st.button("Lancer l'analyse"):
            with st.spinner("Calcul du spectre..."):
                result = spectral_analysis(word, nperseg)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Analyse terminée")
                    st.json(result["results"])
                    fig1, fig2 = result["figures"]
                    st.pyplot(fig1)
                    st.pyplot(fig2)
    else:
        st.info("Aucun mot en mémoire pour l'analyse.")