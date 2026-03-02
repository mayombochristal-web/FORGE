import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
import docx
import pandas as pd
import speech_recognition as sr

# ==========================================
# 1. CONFIGURATION & ADN
# ==========================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
DNA_CORE = "La science est une harmonie. L'esprit cherche la clarté. La nature est le miroir de la vérité."

if not os.path.exists(MEM_DIR):
    os.makedirs(MEM_DIR)
if not os.path.exists(LEXICON_PATH):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

# ==========================================
# 2. LE THALAMUS (Filtrage & Suture)
# ==========================================
def thalamus_processor(text):
    """Filtre les scories et injecte l'ADN pour humaniser le signal brut."""
    if not text: return ""
    # Nettoyage des caractères spéciaux de code
    clean_text = "".join([c for c in text if c.isalnum() or c in " ,.!?\n'"])
    words = clean_text.lower().split()
    
    # Suture automatique : On injecte l'ADN si le texte est trop technique/long
    if len(words) > 50:
        mid = len(words) // 2
        words.insert(mid, DNA_CORE.lower())
    
    return " ".join(words)

# ==========================================
# 3. MOTEUR TTU (Dynamique Interne)
# ==========================================
def evolve_phi(phi, excitation=0.1):
    phi["phi_m"] = max(0.1, min(1.0, phi["phi_m"] + (excitation * 0.2) - 0.01))
    phi["phi_c"] = max(0.1, min(1.0, phi["phi_c"] + (excitation * 0.5) - 0.05))
    phi["phi_d"] = max(0.1, min(1.0, phi["phi_d"] + (excitation * 0.1) - 0.02))
    return phi

# ==========================================
# 4. MÉMOIRE & RÊVE ALPHA
# ==========================================
def load_lex():
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return {}

def save_lex(L):
    # Auto-Homéostasie : Nettoyage auto si saturation (>3000 entrées)
    if len(L) > 3000:
        L = {k: v for k, v in L.items() if len(v) > 1} # On coupe les liens orphelins
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

def learn_with_identity(text, phi, multiplier=1.0):
    text = thalamus_processor(text) # Passage par le Thalamus obligatoire
    words = text.split()
    if len(words) < 2: return
    L = load_lex()
    intensity = math.sqrt(phi["phi_m"]**2 + phi["phi_c"]**2 + phi["phi_d"]**2) * multiplier
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + intensity
    save_lex(L)

def alpha_dream_loop():
    """L'IA génère une pensée interne et l'apprend avec force."""
    phi = st.session_state.phi
    dream = oracle_reply(phi)
    # Si le rêve est cohérent (contient des piliers), on renforce x3
    multiplier = 3.0 if any(word in dream.lower() for word in ["science", "nature", "harmonie"]) else 1.0
    learn_with_identity(dream, phi, multiplier=multiplier)
    return dream

def deep_clean_lexicon(threshold=1.5):
    L = load_lex()
    clean_L = {}
    ban_list = ["uni00a0", "http", "www", "maxiter", "tol=", "data.append"]
    for word, connections in L.items():
        if any(b in word for b in ban_list) or len(word) > 25: continue
        new_conn = {t: w for t, w in connections.items() if w >= threshold}
        if new_conn: clean_L[word] = new_conn
    save_lex(clean_L)
    return len(L) - len(clean_L)

def oracle_reply(phi, seed=None):
    L = load_lex()
    if not L: return "Mémoire vide."
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
    words = [seed]
    for _ in range(int(5 + phi["phi_m"] * 25)):
        current = words[-1]
        if current not in L: break
        options = L[current]
        nxt = max(options, key=options.get) if random.random() > phi["phi_c"] else random.choices(list(options.keys()), weights=list(options.values()))[0]
        words.append(nxt)
        if random.random() < phi["phi_d"] * 0.1: break
    return " ".join(words).capitalize() + "."

# ==========================================
# 5. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="ORACLE V1.5", page_icon="🧠", layout="wide")

if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

st.title("🧠 ORACLE V1.5 : The Alpha Monster")

with st.sidebar:
    st.header("🔬 Diagnostic Cognitif")
    if os.path.exists(LEXICON_PATH):
        st.write(f"Synapses : {len(load_lex())}")
    
    st.divider()
    # Auto-Dream Toggle
    auto_dream = st.toggle("Activer le Rêve Alpha (Éveil)", value=True)
    if auto_dream and random.random() < 0.1: # 10% de chance à chaque interaction
        dream = alpha_dream_loop()
        st.caption(f"💭 Rêve interne : {dream[:50]}...")

    st.divider()
    st.progress(st.session_state.phi["phi_m"], text=f"Masse Φm")
    st.progress(st.session_state.phi["phi_c"], text=f"Chaleur Φc")
    st.progress(st.session_state.phi["phi_d"], text=f"Damping Φd")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion (Via Thalamus)")
    mode = st.radio("Source :", ["Texte", "Document", "Excel", "Audio"])
    raw_content = ""

    if mode == "Texte": raw_content = st.text_area("Entrée :")
    elif mode == "Document":
        file = st.file_uploader("PDF/DOCX", type=["pdf", "docx"])
        if file and file.name.endswith(".pdf"):
            raw_content = " ".join([p.extract_text() for p in PyPDF2.PdfReader(file).pages])
        elif file:
            raw_content = "\n".join([p.text for p in docx.Document(file).paragraphs])
    
    if st.button("⚡ Exciter") and raw_content:
        st.session_state.phi = evolve_phi(st.session_state.phi, 0.4)
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Signal filtré et intégré.")

with col2:
    st.subheader("💬 Sortie de l'Oracle")
    if st.button("Générer Pensée"):
        res = oracle_reply(st.session_state.phi)
        st.info(res)
        # Rumination réflexive
        learn_with_identity(res, {"phi_m":0.1, "phi_c":0.1, "phi_d":0.1}, multiplier=0.5)

st.divider()
if st.button("🌙 Sommeil Profond Automatique"):
    deleted = deep_clean_lexicon(threshold=2.0)
    st.warning(f"Purification terminée : {deleted} liens supprimés.")
    time.sleep(1)
    st.rerun()
