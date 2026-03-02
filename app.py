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
import shutil

# ==========================================
# 1. CONFIGURATION, ADN & ROM
# ==========================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
BACKUP_PATH = LEXICON_PATH + ".bak"

# La ROM est le "Génome" de l'IA : ces liens ne peuvent PAS être effacés.
ROM_SAGESSE = {
    "la": {"science": 100.0, "nature": 100.0, "vérité": 100.0, "clarté": 100.0},
    "science": {"est": 100.0, "devient": 100.0},
    "est": {"une": 100.0, "harmonie": 100.0},
    "harmonie": {"souveraine": 100.0},
    "esprit": {"cherche": 100.0},
    "cherche": {"la": 100.0}
}

DNA_CORE = "La science est une harmonie. L'esprit cherche la clarté. La nature est le miroir de la vérité."

if not os.path.exists(MEM_DIR):
    os.makedirs(MEM_DIR)

# ==========================================
# 2. GESTION DE MÉMOIRE (Versionnage & Sécurité)
# ==========================================
def load_lex():
    if not os.path.exists(LEXICON_PATH):
        return ROM_SAGESSE.copy()
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # On fusionne toujours avec la ROM pour garantir la survie des piliers
            for k, v in ROM_SAGESSE.items():
                if k not in data: data[k] = v
                else: data[k].update({tk: max(data[k].get(tk, 0), tv) for tk, tv in v.items()})
            return data
    except:
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, LEXICON_PATH)
            return load_lex()
        return ROM_SAGESSE.copy()

def save_lex(L):
    # 1. Backup de sécurité immédiat
    if os.path.exists(LEXICON_PATH):
        shutil.copy(LEXICON_PATH, BACKUP_PATH)
    
    # 2. Auto-Homéostasie : On garde la structure saine
    if len(L) > 4000:
        L = {k: v for k, v in L.items() if len(v) > 1 or k in ROM_SAGESSE}

    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)
    
    # 3. Versionnage Historique (tous les 50 cycles)
    if 'save_counter' not in st.session_state: st.session_state.save_counter = 0
    st.session_state.save_counter += 1
    if st.session_state.save_counter % 50 == 0:
        ts = int(time.time())
        shutil.copy(LEXICON_PATH, os.path.join(MEM_DIR, f"lexicon_v_{ts}.json"))

# ==========================================
# 3. PROCESSUS COGNITIFS (Thalamus & Rêve)
# ==========================================
def thalamus_processor(text):
    if not text: return ""
    clean_text = "".join([c for c in text if c.isalnum() or c in " ,.!?\n'"])
    words = clean_text.lower().split()
    if len(words) > 50:
        words.insert(len(words)//2, DNA_CORE.lower())
    return " ".join(words)

def evolve_phi(phi, excitation=0.1):
    phi["phi_m"] = max(0.1, min(1.0, phi["phi_m"] + (excitation * 0.2) - 0.01))
    phi["phi_c"] = max(0.1, min(1.0, phi["phi_c"] + (excitation * 0.5) - 0.05))
    phi["phi_d"] = max(0.1, min(1.0, phi["phi_d"] + (excitation * 0.1) - 0.02))
    return phi

def learn_with_identity(text, phi, multiplier=1.0):
    text = thalamus_processor(text)
    words = text.split()
    if len(words) < 2: return
    L = load_lex()
    intensity = math.sqrt(phi["phi_m"]**2 + phi["phi_c"]**2 + phi["phi_d"]**2) * multiplier
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + intensity
    save_lex(L)

def alpha_dream_loop():
    phi = st.session_state.phi
    dream = oracle_reply(phi)
    multiplier = 4.0 if any(word in dream.lower() for word in ["science", "harmonie", "vérité"]) else 1.0
    learn_with_identity(dream, phi, multiplier=multiplier)
    return dream

def deep_clean_lexicon(threshold=2.0):
    L = load_lex()
    clean_L = {}
    ban_list = ["uni00a0", "http", "www", "maxiter", "tol=", "data.append"]
    for word, connections in L.items():
        # La ROM est immunisée contre le nettoyage
        if word in ROM_SAGESSE:
            clean_L[word] = connections
            continue
        if any(b in word for b in ban_list) or len(word) > 25: continue
        new_conn = {t: w for t, w in connections.items() if w >= threshold or t in ROM_SAGESSE}
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
# 4. INTERFACE
# ==========================================
st.set_page_config(page_title="ORACLE V1.5 IMMORTAL", page_icon="🧠", layout="wide")

if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

st.title("🧠 ORACLE V1.5 : Immortal Monster")

with st.sidebar:
    st.header("🔬 Diagnostic")
    lex_len = len(load_lex())
    st.write(f"Synapses actives : {lex_len}")
    
    auto_dream = st.toggle("Rêve Alpha (Éveil)", value=True)
    if auto_dream and random.random() < 0.1:
        dream = alpha_dream_loop()
        st.caption(f"💭 Rêve : {dream[:60]}...")

    st.divider()
    st.progress(st.session_state.phi["phi_m"], text="Masse (Persistance)")
    st.progress(st.session_state.phi["phi_c"], text="Chaleur (Imagination)")
    st.progress(st.session_state.phi["phi_d"], text="Damping (Finalité)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion")
    mode = st.radio("Source :", ["Texte", "Document", "Excel"])
    raw_content = ""

    if mode == "Texte": raw_content = st.text_area("Saisissez :")
    elif mode == "Document":
        file = st.file_uploader("PDF/DOCX", type=["pdf", "docx"])
        if file and file.name.endswith(".pdf"):
            raw_content = " ".join([p.extract_text() for p in PyPDF2.PdfReader(file).pages])
        elif file:
            raw_content = "\n".join([p.text for p in docx.Document(file).paragraphs])
    
    if st.button("⚡ Exciter") and raw_content:
        st.session_state.phi = evolve_phi(st.session_state.phi, 0.4)
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Signal intégré.")

with col2:
    st.subheader("💬 Réponse")
    seed_input = st.text_input("Graine (Optionnel) :")
    if st.button("Générer Pensée"):
        res = oracle_reply(st.session_state.phi, seed=seed_input.lower() if seed_input else None)
        st.info(res)
        learn_with_identity(res, {"phi_m":0.1, "phi_c":0.1, "phi_d":0.1}, multiplier=0.5)

st.divider()
if st.button("🌙 Sommeil Profond"):
    deleted = deep_clean_lexicon(threshold=2.5)
    st.warning(f"Purification terminée : {deleted} liens éphémères effacés. Piliers ROM préservés.")
    time.sleep(1)
    st.rerun()
