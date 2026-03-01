import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
import docx # Pour Word
import pandas as pd # Pour Excel
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
    """Purifie le lexique et applique le seuil de survie synaptique."""
    L = load_lex()
    if not L: return "Mémoire vide."
    clean_L = {}
    ban_list = ["uni00a0", "http", "www", "....", "____", "........"]

    for word, connections in L.items():
        # Filtre les mots parasites ou trop longs
        if any(b in word for b in ban_list) or len(word) > 30:
            continue
        
        new_connections = {}
        for target, weight in connections.items():
            # Application du THRESHOLD (Seuil de survie)
            if weight >= threshold and not any(b in target for b in ban_list):
                new_connections[target] = weight
        
        if new_connections:
            clean_L[word] = new_connections
            
    save_lex(clean_L)
    return f"Sommeil terminé. {len(L) - len(clean_L)} connexions faibles supprimées."

def oracle_reply(phi, seed=None):
    L = load_lex()
    if not L: return "Mémoire vide. Injectez des données."
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
    M, C, D = phi["phi_m"], phi["phi_c"], phi["phi_d"]
    words = [seed]
    for _ in range(int(5 + M * 25)):
        current = words[-1]
        if current not in L: break
        options = L[current]
        if random.random() > C: 
            nxt = max(options, key=options.get)
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
        words.append(nxt)
        if random.random() < D * 0.1: break
    return " ".join(words).capitalize() + "."

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="ORACLE Autonome", page_icon="🧠", layout="wide")

if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

st.title("🧠 ORACLE : IA Autonome")

# --- SIDEBAR DIAGNOSTIC ET MAINTENANCE ---
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
    seuil_nettoyage = st.slider("Seuil de survie (Threshold)", 1.0, 5.0, 1.5, help="Élimine les liens n'ayant pas atteint ce score d'importance.")
    if st.button("Lancer le Nettoyage"):
        msg = deep_clean_lexicon(threshold=seuil_nettoyage)
        st.warning(msg)
        time.sleep(2)
        st.rerun()

# --- ZONE PRINCIPALE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion de Signal")
    mode = st.radio("Source :", ["Texte", "PDF", "Audio (WAV)"])
    raw_content = ""

    if mode == "Texte":
        raw_content = st.text_area("Entrée :", height=150)
    elif mode == "PDF":
        file = st.file_uploader("Fichier PDF", type="pdf")
        if file:
            reader = PyPDF2.PdfReader(file)
            raw_content = " ".join([p.extract_text() for p in reader.pages])
    elif mode == "Audio (WAV)":
        audio_file = st.file_uploader("Fichier WAV", type="wav")
        if audio_file:
            r = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
                try: raw_content = r.recognize_google(audio_data, language="fr-FR")
                except: st.error("Transcription impossible.")

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
            st.caption("L'Oracle a auto-renforcé cette pensée.")

# --- GESTION DE LA SAUVEGARDE ---
st.divider()
L_actuel = load_lex()
if L_actuel:
    col_a, col_b = st.columns(2)
    with col_a:
        json_brain = json.dumps(L_actuel, indent=2, ensure_ascii=False)
        st.download_button("📥 Exporter le Cerveau (.json)", data=json_brain, file_name="lexicon.json")
    with col_b:
        up = st.file_uploader("📂 Importer un Cerveau", type="json")
        if up:
            save_lex(json.load(up))
            st.success("Mémoire mise à jour.")
