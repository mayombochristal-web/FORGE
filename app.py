import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
import docx  # Pour Word
import pandas as pd  # Pour Excel
import speech_recognition as sr

# ==========================================
# 1. CONFIGURATION & STOCKAGE
# ==========================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

if not os.path.exists(MEM_DIR):
    os.makedirs(MEM_DIR)
if not os.path.exists(LEXICON_PATH):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

# ==========================================
# 2. MOTEUR TTU (Dynamique Interne)
# ==========================================
def evolve_phi(phi, excitation=0.1):
    phi["phi_m"] = max(0.1, min(1.0, phi["phi_m"] + (excitation * 0.2) - 0.01))
    phi["phi_c"] = max(0.1, min(1.0, phi["phi_c"] + (excitation * 0.5) - 0.05))
    phi["phi_d"] = max(0.1, min(1.0, phi["phi_d"] + (excitation * 0.1) - 0.02))
    return phi

# ==========================================
# 3. MÉMOIRE & APPRENTISSAGE (Auto-Worker)
# ==========================================
def load_lex():
    if os.path.exists(LEXICON_PATH):
        try:
            with open(LEXICON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_lex(L):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

def learn_with_identity(text, phi):
    words = text.lower().split()
    if len(words) < 2: return
    L = load_lex()
    intensity = math.sqrt(phi["phi_m"]**2 + phi["phi_c"]**2 + phi["phi_d"]**2)
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + intensity
    save_lex(L)

def deep_clean_lexicon(threshold=1.5):
    L = load_lex()
    if not L: return "Mémoire vide."
    clean_L = {}
    ban_list = ["uni00a0", "http", "www", "....", "____", "........"]
    for word, connections in L.items():
        if any(b in word for b in ban_list) or len(word) > 30:
            continue
        new_connections = {t: w for t, w in connections.items() if w >= threshold and not any(b in t for b in ban_list)}
        if new_connections:
            clean_L[word] = new_connections
    save_lex(clean_L)
    return f"Sommeil terminé. {len(L) - len(clean_L)} nœuds nettoyés."

def oracle_reply(phi, seed=None):
    L = load_lex()
    if not L: return "Mémoire vide. Injectez des données."
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
    words = [seed]
    for _ in range(int(5 + phi["phi_m"] * 25)):
        current = words[-1]
        if current not in L: break
        options = L[current]
        if random.random() > phi["phi_c"]: 
            nxt = max(options, key=options.get)
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
        words.append(nxt)
        if random.random() < phi["phi_d"] * 0.1: break
    return " ".join(words).capitalize() + "."

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="ORACLE Autonome", page_icon="🧠", layout="wide")

if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

st.title("🧠 ORACLE : IA Autonome Multi-Source")

with st.sidebar:
    st.header("🛠 État du Système")
    if os.path.exists(LEXICON_PATH):
        taille = os.path.getsize(LEXICON_PATH) / 1024
        st.success(f"Mémoire : {taille:.2f} KB")
    
    st.divider()
    st.subheader("📊 Dynamique Φ")
    st.progress(st.session_state.phi["phi_m"], text=f"Mémoire (M) : {st.session_state.phi['phi_m']:.2f}")
    st.progress(st.session_state.phi["phi_c"], text=f"Créativité (C) : {st.session_state.phi['phi_c']:.2f}")
    st.progress(st.session_state.phi["phi_d"], text=f"Stabilité (D) : {st.session_state.phi['phi_d']:.2f}")

    st.divider()
    st.subheader("🌙 Sommeil Profond")
    seuil = st.slider("Seuil de survie (Threshold)", 1.0, 5.0, 1.5)
    if st.button("Lancer le Nettoyage"):
        msg = deep_clean_lexicon(threshold=seuil)
        st.warning(msg)
        time.sleep(1)
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion de Signal")
    mode = st.radio("Source :", ["Texte", "Document (PDF/Word/TXT)", "Excel", "Audio (WAV)"])
    raw_content = ""

    if mode == "Texte":
        raw_content = st.text_area("Entrée :", height=150)
    
    elif mode == "Document (PDF/Word/TXT)":
        file = st.file_uploader("Fichier", type=["pdf", "docx", "txt"])
        if file:
            if file.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file)
                raw_content = " ".join([p.extract_text() for p in reader.pages])
            elif file.name.endswith(".docx"):
                doc = docx.Document(file)
                raw_content = " ".join([p.text for p in doc.paragraphs])
            else:
                raw_content = file.read().decode("utf-8")

    elif mode == "Excel":
        file = st.file_uploader("Tableur", type=["xlsx", "xls"])
        if file:
            df = pd.read_excel(file)
            raw_content = df.to_string()

    elif mode == "Audio (WAV)":
        audio_file = st.file_uploader("Audio", type="wav")
        if audio_file:
            r = sr.Recognizer()
            with sr.AudioFile(audio_file) as src:
                try: raw_content = r.recognize_google(r.record(src), language="fr-FR")
                except: st.error("Échec transcription.")

    if st.button("⚡ Exciter l'Oracle") and raw_content:
        st.session_state.phi = evolve_phi(st.session_state.phi, 0.4)
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Apprentissage réussi.")

with col2:
    st.subheader("💬 Réponse")
    if st.button("Générer une Pensée"):
        res = oracle_reply(st.session_state.phi)
        st.info(res)
        if random.random() < 0.3:
            learn_with_identity(res, {"phi_m":0.1, "phi_c":0.1, "phi_d":0.1})
            st.caption("Auto-renforcement activé.")

st.divider()
L_actuel = load_lex()
if L_actuel:
    c_a, c_b = st.columns(2)
    with c_a:
        st.download_button("📥 Exporter Lexicon", data=json.dumps(L_actuel, indent=2, ensure_ascii=False), file_name="lexicon.json")
    with c_b:
        up = st.file_uploader("📂 Importer Lexicon", type="json")
        if up:
            save_lex(json.load(up))
            st.rerun()
