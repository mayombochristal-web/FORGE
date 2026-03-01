import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
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
        json.dump({}, f) # Crée un cerveau vide

# ==========================================
# 2. MOTEUR TTU (Dynamique Interne)
# ==========================================
def evolve_phi(phi, excitation=0.1):
    """Fait évoluer l'état dynamique Φ en fonction des entrées."""
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
        json.dump(L, f, indent=2)

def learn_with_identity(text, phi):
    """Apprentissage pondéré par l'état Φ (Identité)."""
    words = text.lower().split()
    if len(words) < 2: return
    L = load_lex()
    
    # Intensité de l'ancrage mémoire
    intensity = math.sqrt(phi["phi_m"]**2 + phi["phi_c"]**2 + phi["phi_d"]**2)
    
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        # On renforce le lien entre le mot A et le mot B
        L[a][b] = L[a].get(b, 0) + intensity
    save_lex(L)

def oracle_reply(phi, seed=None):
    """Génération probabiliste guidée par Φ."""
    L = load_lex()
    if not L: return "Mémoire vide. Injectez des données (Audio, PDF, Texte)."
    
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
        
    M, C, D = phi["phi_m"], phi["phi_c"], phi["phi_d"]
    words = [seed]
    
    # Longueur de phrase influencée par Φ_M
    for _ in range(int(5 + M * 25)):
        current = words[-1]
        if current not in L: break
        options = L[current]
        
        # Logique d'identité : Habitude vs Créativité
        if random.random() > C: 
            nxt = max(options, key=options.get) # Chemin le plus fort
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
            
        words.append(nxt)
        if random.random() < D * 0.1: break # Stabilité
        
    return " ".join(words).capitalize() + "."

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="ORACLE Autonome", page_icon="🧠", layout="wide")

# Initialisation de l'état
if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}

st.title("🧠 ORACLE : IA Autonome sans LLM")
st.markdown("---")

# Barre latérale : Moniteur de conscience
with st.sidebar:
    st.header("📊 État Interne Φ")
    st.progress(st.session_state.phi["phi_m"], text=f"Masse (Mémoire) : {st.session_state.phi['phi_m']:.2f}")
    st.progress(st.session_state.phi["phi_c"], text=f"Chaleur (Créativité) : {st.session_state.phi['phi_c']:.2f}")
    st.progress(st.session_state.phi["phi_d"], text=f"Damping (Stabilité) : {st.session_state.phi['phi_d']:.2f}")
    
    if st.button("🌙 Cycle de Sommeil (Oubli)"):
        L = load_lex()
        # Évaporation de 15% et suppression des liens morts
        L = {w: {t: p*0.85 for t, p in c.items() if p*0.85 > 0.1} for w, c in L.items()}
        save_lex(L)
        st.success("Mémoire consolidée.")

# Zone d'Entrée Multimodale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion Universelle")
    mode = st.radio("Type de signal :", ["Texte", "PDF", "Audio (WAV)"])
    raw_content = ""

    if mode == "Texte":
        raw_content = st.text_area("Saisissez votre texte :", height=150)
    elif mode == "PDF":
        file = st.file_uploader("Charger un PDF", type="pdf")
        if file:
            reader = PyPDF2.PdfReader(file)
            raw_content = " ".join([p.extract_text() for p in reader.pages])
    elif mode == "Audio (WAV)":
        audio_file = st.file_uploader("Charger un audio", type="wav")
        if audio_file:
            r = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
                try: raw_content = r.recognize_google(audio_data, language="fr-FR")
                except: st.error("Échec de la transcription.")

    if st.button("⚡ Exciter l'Oracle") and raw_content:
        st.session_state.phi = evolve_phi(st.session_state.phi, 0.4)
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Signal intégré au lexique identitaire.")

with col2:
    st.subheader("💬 Réponse de l'Oracle")
    if st.button("Générer une Pensée"):
        with st.spinner("Rumination..."):
            res = oracle_reply(st.session_state.phi)
            st.markdown(f"> **{res}**")
            
            # Auto-renforcement (Rumination)
            if random.random() < 0.3:
                learn_with_identity(res, {"phi_m":0.1, "phi_c":0.1, "phi_d":0.1})
                st.caption("ℹ️ L'Oracle vient d'auto-apprendre de sa propre pensée.")

st.divider()
st.caption("Système autonome basé sur des chaînes de Markov pondérées par des systèmes dynamiques Φ.")
