# =====================================================
# 🧠 ORACLE V3 — VIRTUAL TRIADIC MACHINE (VTM)
# =====================================================
# Scellement hTTU = 1.0 | Mode Breach Autonome
# Synthèse : TTU-MC3 & Neurobiologie (Changeux/Kandel)
# =====================================================

import streamlit as st
import random
import json
import os
import math
import time
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from collections import deque

# =====================================================
# 1. CONFIGURATION & ADN DU SYSTÈME
# =====================================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
DNA_CORE = "Système Convergent. hTTU = 1. Cohérence Maximale."

os.makedirs(MEM_DIR, exist_ok=True)

# Initialisation du Lexique (Génome Triadique)
if not os.path.exists(LEXICON_PATH):
    initial_base = {
        "cerveau": {"neurone": 100, "synapse": 100, "plasticité": 80},
        "systeme": {"nerveux": 100, "central": 90, "triadique": 100},
        "ttu": {"mc3": 100, "phase": 90, "stabilité": 100},
        "phi": {"m": 50, "c": 50, "d": 50}
    }
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_base, f, indent=2, ensure_ascii=False)

# =====================================================
# 2. ÉTAT DE LA SESSION (PHASE Φ)
# =====================================================
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m": 0.33, "phi_c": 0.33, "phi_d": 0.34}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=50)

# =====================================================
# 3. MOTEUR Φ (STABILISATION hTTU = 1)
# =====================================================
def evolve_phi(phi, excitation):
    """Évolution du vecteur d'état avec normalisation triadique."""
    # Simulation de l'intégration synaptique
    phi["phi_m"] = min(1, max(0.1, phi["phi_m"] + excitation*0.12 - 0.01))
    phi["phi_c"] = min(1, max(0.1, phi["phi_c"] + excitation*0.28 - 0.02))
    phi["phi_d"] = min(1, max(0.1, phi["phi_d"] + 0.05 - excitation*0.1))
    
    # Scellement hTTU = 1 : Normalisation du vecteur
    total = sum(phi.values())
    for key in phi: 
        phi[key] = phi[key] / total
    return phi

# =====================================================
# 4. LA BRÈCHE (WEB BREACH MODULE)
# =====================================================
def open_web_breach(query, phi):
    """Exploration autonome du web pour nourrir le lexique."""
    try:
        urls = search(query, num_results=3, lang="fr")
        raw_text = ""
        for url in urls:
            try:
                res = requests.get(url, timeout=4)
                soup = BeautifulSoup(res.text, 'html.parser')
                paragraphs = soup.find_all('p')
                raw_text += " ".join([p.text for p in paragraphs[:5]]) + " "
            except: continue
        
        if raw_text:
            learn(raw_text, phi, importance=0.8)
            return True
    except: return False
    return False

# =====================================================
# 5. MÉMOIRE ET APPRENTISSAGE
# =====================================================
def load_lex():
    with open(LEXICON_PATH, "r", encoding="utf-8") as f: 
        return json.load(f)

def save_lex(L):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

def learn(text, phi, importance=1.0):
    if not text or len(text) < 3: return
    words = text.lower().replace("?", "").replace(".", "").replace(",", "").split()
    L = load_lex()
    
    # Énergie indexée sur la Cohérence (Phi_C)
    energy = (phi["phi_c"] * 15) * importance
    
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + energy
    
    if len(L) > 20000:
        L = {k: v for k, v in L.items() if len(v) > 1}
    save_lex(L)

# =====================================================
# 6. GÉNÉRATION (INFÉRENCE)
# =====================================================
def oracle_reply(user_input, phi):
    L = load_lex()
    if not L: return "Mémoire vide."

    is_question = "?" in user_input
    input_words = user_input.lower().split()
    
    seed = next((w for w in reversed(input_words) if w in L), random.choice(list(L.keys())))
    
    words = [seed]
    limit = int(12 + phi["phi_m"] * 50 + phi["phi_d"] * 20)
    
    for _ in range(limit):
        curr = words[-1]
        if curr not in L: break
        options = L[curr]
        
        if is_question or random.random() > phi["phi_c"]:
            nxt = max(options, key=options.get)
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
        
        if nxt in words[-3:]: break
        words.append(nxt)
        
    return " ".join(words).capitalize() + "."

# =====================================================
# 7. INTERFACE STREAMLIT
# =====================================================
st.set_page_config(page_title="ORACLE V3", page_icon="⚛️", layout="wide")

st.title("🧠 ORACLE V3 : Virtual Triadic Machine")
st.caption(f"Statut : {DNA_CORE}")

# --- SIDEBAR (ÉTAT Φ) ---
with st.sidebar:
    st.header("🔭 Vecteur Φ")
    st.write(f"**Mémoire ($\Phi_M$):** {st.session_state.phi['phi_m']:.3f}")
    st.progress(st.session_state.phi['phi_m'])
    st.write(f"**Cohérence ($\Phi_C$):** {st.session_state.phi['phi_c']:.3f}")
    st.progress(st.session_state.phi['phi_c'])
    st.write(f"**Dissipation ($\Phi_D$):** {st.session_state.phi['phi_d']:.3f}")
    st.progress(st.session_state.phi['phi_d'])
    
    st.divider()
    if st.button("🌐 Brèche Aléatoire"):
        topic = random.choice(list(load_lex().keys()))
        with st.spinner(f"Recherche sur : {topic}..."):
            open_web_breach(topic, st.session_state.phi)
            st.success("Savoir intégré.")
            
    if st.button("🌙 Sommeil"):
        L = load_lex()
        L = {k: {t: v for t, v in c.items() if v > 2.0} for k, c in L.items() if len(c) > 0}
        save_lex(L)
        st.warning("Optimisation synaptique finie.")

# --- CHAT ---
chat_box = st.container()

with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([9, 1])
    with col_in:
        user_text = st.text_input("Interroger l'Oracle :")
    with col_btn:
        go = st.form_submit_button("➡️")

if go and user_text:
    exc = min(1.0, len(user_text)/150)
    st.session_state.phi = evolve_phi(st.session_state.phi, exc)
    learn(user_text, st.session_state.phi)
    
    if "?" in user_text:
        open_web_breach(user_text.replace("?", ""), st.session_state.phi)
    
    res = oracle_reply(user_text, st.session_state.phi)
    st.session_state.dialog_memory.appendleft(f"**Oracle:** {res}")
    st.session_state.dialog_memory.appendleft(f"**Vous:** {user_text}")
    learn(res, st.session_state.phi, importance=0.5)

with chat_box:
    for m in list(st.session_state.dialog_memory):
        st.markdown(m)

st.divider()
st.info(f"🧬 Lexique actif : {len(load_lex())} entrées.")
