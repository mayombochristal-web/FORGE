import streamlit as st
import numpy as np
import pandas as pd
import json, os, math, time, re, datetime
import random
from collections import deque, Counter
import matplotlib.pyplot as plt
from scipy.signal import stft

# =========================================================
# CONFIGURATION ET STYLE (THÈME SOMBRE SCIENTIFIQUE)
# =========================================================
st.set_page_config(page_title="ORACLE Ω-TTU v5.0", layout="wide", page_icon="⚛️")

st.markdown("""
    <style>
    .reportview-container { background: #0a0a0f; }
    .stMetric { background: rgba(30, 30, 50, 0.5); border-radius: 10px; padding: 15px; border: 1px solid #3e3e5e; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# NOYAU TTU-MC³ (LOGIQUE PHYSIQUE)
# =========================================================
def normalize_ttu(m, c, d):
    norm = math.sqrt(m**2 + c**2 + d**2) + 1e-9
    return m/norm, c/norm, d/norm

def evolve_ttu(phi, excitation):
    # Logique de la Station d'Audit : Transition de phase
    new_m = min(1.0, max(0.1, phi["phi_m"] + excitation * 0.12 - 0.02))
    new_c = min(1.0, max(0.1, phi["phi_c"] + excitation * 0.25 - 0.04))
    new_d = min(1.0, max(0.1, phi["phi_d"] + 0.05 - excitation * 0.08))
    
    m, c, d = normalize_ttu(new_m, new_c, new_d)
    return {"phi_m": m, "phi_c": c, "phi_d": d}

# =========================================================
# GESTION DE LA MÉMOIRE (TST GHOST MEMORY)
# =========================================================
MEM_FILE = "ttu_memory.json"

def load_memory():
    if not os.path.exists(MEM_FILE): return {}
    with open(MEM_FILE, "r", encoding="utf-8") as f: return json.load(f)

def save_memory(data):
    with open(MEM_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

# =========================================================
# INITIALISATION SESSION STATE
# =========================================================
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}
if "history" not in st.session_state:
    st.session_state.history = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []

# =========================================================
# MOTEUR DE GÉNÉRATION (RECONSTITUTION DE RUPTURE)
# =========================================================
def generate_ttu_text(prompt, length=40):
    mem = load_memory()
    if not mem: return "Mémoire vide. Injectez des données pour initialiser l'espace TTU."
    
    words = re.sub(r"[^a-zàéèêâîôû\s]", "", prompt.lower()).split()
    seed = words[-1] if words and words[-1] in mem else random.choice(list(mem.keys()))
    
    sentence = [seed.capitalize()]
    current = seed
    
    for _ in range(length):
        if current not in mem: break
        
        candidates = mem[current]
        # Choix basé sur la Cohérence (phi_c) vs Aléatoire (phi_d)
        if random.random() < st.session_state.phi["phi_c"]:
            next_word = max(candidates, key=candidates.get) # Chemin le plus probable
        else:
            # Bifurcation de Morse-Smale (choix probabiliste)
            choices = list(candidates.keys())
            weights = list(candidates.values())
            next_word = random.choices(choices, weights=weights)[0]
            
        sentence.append(next_word)
        current = next_word
        
        # Dissipation : Sortie de boucle si répétition excessive
        if len(sentence) > 10 and sentence[-1] == sentence[-3]:
            if random.random() < st.session_state.phi["phi_d"]: break
            
    return " ".join(sentence) + "."

# =========================================================
# INTERFACE UTILISATEUR (UX MODERNE)
# =========================================================
st.title("⚛️ ORACLE Ω-TTU v5.0")
st.caption("Station d'Audit Linguistique par Projection Triadique et Analyse Spectrale")

with st.sidebar:
    st.header("🔬 Diagnostic TTU")
    p = st.session_state.phi
    st.metric("Mémoire (Φm)", f"{p['phi_m']:.2f}")
    st.progress(p['phi_m'])
    st.metric("Cohérence (Φc)", f"{p['phi_c']:.2f}")
    st.progress(p['phi_c'])
    st.metric("Dissipation (Φd)", f"{p['phi_d']:.2f}")
    st.progress(p['phi_d'])
    
    st.divider()
    if st.button("🌙 Cycle de Sommeil"):
        mem = load_memory()
        # Nettoyage entropique : on supprime les connexions < 1.2 (Dissipation)
        new_mem = {w: {k: v for k, v in c.items() if v > 1.2} for w, c in mem.items() if len(c) > 0}
        save_memory(new_mem)
        st.success("Entropie réduite.")

# ZONE D'AFFICHAGE CHAT
chat_container = st.container()
with chat_container:
    for chat in st.session_state.history:
        role = "👤" if chat["role"] == "user" else "⚛️"
        st.markdown(f"**{role} :** {chat['content']}")

# INPUT UTILISATEUR
with st.container():
    st.divider()
    c1, c2 = st.columns([8, 1])
    user_input = c1.text_input("Saisissez une intention ou injectez un signal...", placeholder="Écrivez ici...")
    send_btn = c2.button("Envoyer")
    
    uploaded_file = st.file_uploader("📥 Injecter Document (Audit)", type=["txt", "pdf"])

if send_btn or user_input:
    if user_input:
        # 1. Phase d'Excitation
        excitation = min(1.0, len(user_input) / 100)
        st.session_state.phi = evolve_ttu(st.session_state.phi, excitation)
        
        # 2. Apprentissage (TST)
        mem = load_memory()
        tokens = re.sub(r"[^a-zàéèêâîôû\s]", "", user_input.lower()).split()
        for i in range(len(tokens)-1):
            a, b = tokens[i], tokens[i+1]
            mem.setdefault(a, {})
            mem[a][b] = mem[a].get(b, 0) + (1.0 + st.session_state.phi["phi_m"])
        save_memory(mem)
        
        # 3. Génération (TTU)
        reply = generate_ttu_text(user_input)
        
        # Mise à jour historique
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "oracle", "content": reply})
        st.rerun()

# =========================================================
# ANALYSE SPECTRALE (RETOUR VISUEL)
# =========================================================
if st.checkbox("📊 Afficher l'Analyse Spectrale du Signal"):
    st.subheader("Analyse Spectrale TST (Transition de Phase)")
    mem = load_memory()
    if mem:
        all_words = list(mem.keys())
        word_to_plot = st.selectbox("Sélectionner un concept à auditer", all_words)
        
        # Création d'un signal factice basé sur les poids en mémoire pour la démo
        weights = list(mem[word_to_plot].values()) if word_to_plot in mem else [0]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(weights, color='#00f2fe', lw=2)
        ax.fill_between(range(len(weights)), weights, color='#4facfe', alpha=0.3)
        ax.set_title(f"Signature Energétique de '{word_to_plot}'")
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        st.pyplot(fig)
    else:
        st.info("Aucune donnée en mémoire pour l'audit.")
