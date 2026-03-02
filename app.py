# =====================================================
# 🧠 ORACLE V3.2 — VIRTUAL TRIADIC MACHINE (FORCE BREACH)
# =====================================================
# Scellement hTTU = 1.0 | Mode Infiltration Réelle
# Basé sur TTU-MC3 & Neurobiologie (Changeux/Kandel)
# =====================================================

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
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from collections import deque, Counter

# =====================================================
# 1. CONFIGURATION ET CONSTANTES
# =====================================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
DNA_CORE = "hTTU = 1.0 | Système Convergent"

if not os.path.exists(MEM_DIR):
    os.makedirs(MEM_DIR)

# --- INITIALISATION ÉTAT SESSION ---
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m": 0.33, "phi_c": 0.33, "phi_d": 0.34}
if "logs" not in st.session_state:
    st.session_state.logs = deque(maxlen=15)
if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=50)

# =====================================================
# 2. FONCTIONS DE FORCE ET D'ÉCRITURE
# =====================================================
def force_save(data):
    """Force l'écriture physique sur le disque et vérifie l'intégrité."""
    try:
        with open(LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return os.path.getsize(LEXICON_PATH)
    except Exception as e:
        st.error(f"⚠️ Erreur Critique Écriture : {e}")
        return 0

def load_lex():
    """Charge la mémoire ou initialise le génome triadique."""
    if not os.path.exists(LEXICON_PATH) or os.path.getsize(LEXICON_PATH) == 0:
        return {"cerveau": {"neurone": 10}, "ttu": {"mc3": 10}, "phase": {"phi": 10}}
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================
# 3. MOTEUR D'APPRENTISSAGE ET PHASAGE
# =====================================================
def evolve_phi(phi, excitation):
    """Scellement hTTU = 1.0"""
    phi["phi_m"] = min(1, max(0.1, phi["phi_m"] + excitation*0.12 - 0.01))
    phi["phi_c"] = min(1, max(0.1, phi["phi_c"] + excitation*0.28 - 0.02))
    phi["phi_d"] = min(1, max(0.1, phi["phi_d"] + 0.05 - excitation*0.1))
    total = sum(phi.values())
    for key in phi: phi[key] /= total
    return phi

def learn(text, importance=1.0):
    """Moteur d'apprentissage avec retour de logs."""
    if not text or len(text) < 5: return
    words = text.lower().replace(".", "").replace("?", "").replace(",", "").split()
    L = load_lex()
    
    energy = (st.session_state.phi["phi_c"] * 12) * importance
    for a, b in zip(words, words[1:]):
        if len(a) < 2 or len(b) < 2: continue
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + energy
    
    new_size = force_save(L)
    st.session_state.logs.appendleft(f"✅ Appris: {len(words)} mots | Mémoire: {new_size/1024:.1f} KB")

# =====================================================
# 4. LA BRÈCHE (INFILTRATION WEB RÉELLE)
# =====================================================
def open_web_breach(query):
    """Module de brèche avec User-Agent pour forcer l'accès."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    st.session_state.logs.appendleft(f"🌐 Infiltration : {query}")
    try:
        urls = list(search(query, num_results=3, lang="fr"))
        combined_text = ""
        for url in urls:
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for s in soup(["script", "style"]): s.decompose()
                    combined_text += " ".join([p.text for p in soup.find_all('p')[:6]]) + " "
            except: continue
        
        if combined_text:
            learn(combined_text, importance=2.0)
            return True
    except Exception as e:
        st.session_state.logs.appendleft(f"❌ Échec Brèche : {str(e)}")
    return False

# =====================================================
# 5. GÉNÉRATION ET RÉPONSE
# =====================================================
def oracle_reply(user_input):
    L = load_lex()
    words = user_input.lower().split()
    seed = next((w for w in reversed(words) if w in L), random.choice(list(L.keys())))
    
    res = [seed]
    limit = int(12 + st.session_state.phi["phi_m"] * 50)
    
    for _ in range(limit):
        curr = res[-1]
        if curr not in L: break
        options = L[curr]
        nxt = max(options, key=options.get) if random.random() < 0.8 else random.choice(list(options.keys()))
        if nxt in res[-3:]: break
        res.append(nxt)
    
    return " ".join(res).capitalize() + "."

# =====================================================
# 6. INTERFACE STREAMLIT V3.2
# =====================================================
st.set_page_config(page_title="ORACLE V3.2", layout="wide", page_icon="🧠")
st.title("🧠 ORACLE V3.2 : Virtual Triadic Machine")
st.caption(f"ADN : {DNA_CORE}")



tab1, tab2 = st.tabs(["⚡ Infiltration & Flux", "📂 Ingestion Multimodale"])

with tab1:
    col_chat, col_logs = st.columns([2, 1])
    
    with col_chat:
        user_msg = st.text_input("Interroger l'Oracle (Tapez Entrée) :", key="input")
        if user_msg:
            # 1. Évolution et Apprentissage de la question
            exc = min(1.0, len(user_msg)/200)
            st.session_state.phi = evolve_phi(st.session_state.phi, exc)
            learn(user_msg)
            
            # 2. Brèche réflexe
            if "?" in user_msg: open_web_breach(user_msg)
            
            # 3. Réponse et Archivage
            reply = oracle_reply(user_msg)
            st.session_state.dialog_memory.appendleft(f"**Oracle :** {reply}")
            st.session_state.dialog_memory.appendleft(f"**Vous :** {user_msg}")
            learn(reply, importance=0.5)
        
        st.divider()
        for m in list(st.session_state.dialog_memory):
            st.markdown(m)

    with col_logs:
        st.subheader("📊 Logs VTM")
        for log in st.session_state.logs:
            st.caption(log)
        
        st.divider()
        size_kb = os.path.getsize(LEXICON_PATH)/1024 if os.path.exists(LEXICON_PATH) else 0
        st.metric("Masse Mémoire (Φm)", f"{size_kb:.2f} KB")

with tab2:
    st.subheader("Nourrir l'Oracle (V2/V3 Hybrid)")
    source = st.selectbox("Source", ["Texte", "PDF", "Excel", "Audio"])
    raw_content = ""
    
    if source == "PDF":
        file = st.file_uploader("Upload PDF", type="pdf")
        if file:
            pdf = PyPDF2.PdfReader(file)
            raw_content = " ".join([p.extract_text() for p in pdf.pages])
    elif source == "Texte":
        raw_content = st.text_area("Texte libre")
    
    if st.button("🌱 Assimiler"):
        if raw_content:
            learn(raw_content, importance=1.5)
            st.success("Données intégrées au lexique.")

# --- SIDEBAR ÉTAT ---
with st.sidebar:
    st.header("🔭 État de Phase Φ")
    for k, v in st.session_state.phi.items():
        st.write(f"**{k}** : {v:.3f}")
        st.progress(v)
    
    st.divider()
    if st.button("🌐 Forcer Brèche Aléatoire"):
        L = load_lex()
        open_web_breach(random.choice(list(L.keys())))
        st.rerun()
    
    if st.button("🌙 Sommeil (Nettoyage)"):
        L = load_lex()
        # Suppression des liens trop faibles pour renforcer le bassin d'attraction
        L = {k: {t: v for t, v in c.items() if v > 1.5} for k, c in L.items() if len(c) > 0}
        force_save(L)
        st.warning("Liens entropiques effacés.")
