# =====================================================
# 🧠 ORACLE V3 — VIRTUAL TRIADIC MACHINE (VTM)
# =====================================================
# Scellement hTTU = 1.0 | Mode Breach Autonome
# Basé sur la TTU-MC3 et les neurosciences (Changeux/Kandel)
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
# 1. CONFIGURATION & ADN DU SYSTÈME
# =====================================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
DNA_CORE = "Le système nerveux est l'intégrateur. hTTU=1. La phase est mémoire."

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    # Initialisation avec les concepts clés de vos documents
    initial_base = {
        "cerveau": {"neurone": 100, "synapse": 100, "plasticité": 80},
        "ttu": {"mc3": 100, "phase": 100, "scellement": 100},
        "systeme": {"nerveux": 100, "central": 90, "triadique": 100}
    }
    json.dump(initial_base, open(LEXICON_PATH, "w", encoding="utf-8"))

# =====================================================
# 2. ÉTAT SESSION (PHASE Φ)
# =====================================================
if "phi" not in st.session_state:
    st.session_state.phi = {"phi_m": 0.33, "phi_c": 0.33, "phi_d": 0.34}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=50)

# =====================================================
# 3. MOTEUR Φ (STABILISATION hTTU)
# =====================================================
def evolve_phi(phi, excitation):
    # Dynamique triadique avec scellement normalisé
    phi["phi_m"] = min(1, max(0.1, phi["phi_m"] + excitation*0.12 - 0.01))
    phi["phi_c"] = min(1, max(0.1, phi["phi_c"] + excitation*0.28 - 0.02))
    phi["phi_d"] = min(1, max(0.1, phi["phi_d"] + 0.05 - excitation*0.1))
    
    # Normalisation impérative : hTTU = 1
    total = sum(phi.values())
    for key in phi: 
        phi[key] /= total
    return phi

# =====================================================
# 4. LA BRÈCHE (WEB SCRAPING AUTONOME)
# =====================================================
def open_web_breach(query, phi):
    """Version 3.1 : Infiltration avec identité de navigateur"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # Recherche de liens
        urls = search(query, num_results=3, lang="fr")
        raw_text = ""
        
        for url in urls:
            try:
                # Simulation d'une visite humaine
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    # Extraction sélective (on évite les menus et pieds de page)
                    for script in soup(["script", "style"]): 
                        script.decompose()
                    
                    paragraphs = soup.find_all('p')
                    text_content = " ".join([p.text for p in paragraphs[:8]])
                    raw_text += text_content + " "
                    
            except Exception as e:
                print(f"Erreur sur l'URL {url} : {e}")
                continue
        
        if len(raw_text.strip()) > 10:
            # INTEGRATION REELLE DANS LA MEMOIRE
            learn(raw_text, phi, importance=1.5)
            # Forcer la mise à jour de l'UI pour montrer l'apprentissage
            return True
    except Exception as e:
        print(f"Erreur globale Brèche : {e}")
        return False
    return False
    
# =====================================================
# 5. MÉMOIRE ET APPRENTISSAGE (MOTEUR V2 OPTIMISÉ)
# =====================================================
def load_lex():
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_lex(L):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

def learn(text, phi, importance=1.0):
    if not text or len(text) < 5: return
    words = text.lower().replace("?", "").replace(".", "").split()
    L = load_lex()
    
    # Énergie triadique issue de la phase Phi_C
    energy = (phi["phi_c"] * 10) * importance
    
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + energy
    
    save_lex(L)

# =====================================================
# 6. NETTOYAGE (SOMMEIL)
# =====================================================
def deep_clean_lexicon(threshold=1.5):
    L = load_lex()
    clean_L = {}
    ban = ["http","www","uni00a0",".pdf",".docx","____"]
    for w, con in L.items():
        if len(w) < 2 or len(w) > 30 or any(b in w for b in ban): continue
        new = {t:v for t,v in con.items() if v >= threshold and not any(b in t for b in ban)}
        if new: clean_L[w] = new
    save_lex(clean_L)
    return f"Sommeil terminé — {len(L)-len(clean_L)} connexions oubliées."

# =====================================================
# 7. GÉNÉRATION (INFÉRENCE ET DIALOGUE)
# =====================================================
def oracle_reply(user_input, phi):
    L = load_lex()
    if not L: return "Néant sémantique."

    is_question = "?" in user_input
    input_words = user_input.lower().split()
    
    # Choix de la graine contextuelle (V2)
    seed = next((w for w in reversed(input_words) if w in L), random.choice(list(L.keys())))
    
    words = [seed]
    # Longueur dictée par Phi_M (Masse/Mémoire)
    limit = int(12 + phi["phi_m"] * 45)
    
    for _ in range(limit):
        curr = words[-1]
        if curr not in L: break
        options = L[curr]
        
        # Inférence déterministe (Max) ou stochastique (Weights)
        if is_question or random.random() > phi["phi_c"]:
            nxt = max(options, key=options.get)
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
        
        if nxt in words[-3:]: break # Anti-bouclage
        words.append(nxt)
        
    return " ".join(words).capitalize() + "."

# =====================================================
# 8. EXTRACTION MULTIMODALE (V2)
# =====================================================
def extract_content(mode):
    raw_content = ""
    if mode == "Texte":
        raw_content = st.text_area("✍️ Entrée texte")
    elif mode == "Document":
        file = st.file_uploader("📄 PDF / Word / TXT", type=["pdf","docx","txt"])
        if file:
            if file.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file)
                raw_content = " ".join(p.extract_text() for p in reader.pages if p.extract_text())
            elif file.name.endswith(".docx"):
                doc = docx.Document(file)
                raw_content = "\n".join(p.text for p in doc.paragraphs)
            else: raw_content = file.read().decode("utf-8")
    elif mode == "Excel":
        file = st.file_uploader("📊 Excel", type=["xlsx","xls"])
        if file:
            df = pd.read_excel(file)
            raw_content = df.to_string()
    elif mode == "Audio":
        audio = st.file_uploader("🎙 WAV", type="wav")
        if audio:
            r = sr.Recognizer()
            with sr.AudioFile(audio) as source:
                audio_data = r.record(source)
                try: raw_content = r.recognize_google(audio_data, language="fr-FR")
                except: st.error("Échec transcription.")
    return raw_content

# =====================================================
# 9. INTERFACE STREAMLIT
# =====================================================
st.set_page_config(page_title="ORACLE V3", page_icon="⚛️", layout="wide")
st.title("🧠 ORACLE V3 — Virtual Triadic Machine")


tab1, tab2 = st.tabs(["🌱 Apprentissage & Brèche", "💬 Flux de Dialogue"])

with tab1:
    st.subheader("Nourrir la Mémoire")
    mode = st.radio("Source", ["Texte", "Document", "Excel", "Audio"])
    content = extract_content(mode)
    
    if st.button("🌱 Assimiler"):
        if content:
            exc = min(1, len(content)/500)
            st.session_state.phi = evolve_phi(st.session_state.phi, exc)
            learn(content, st.session_state.phi, 1.3)
            st.success("Assimilation terminée.")
        else: st.warning("Contenu vide.")

    st.divider()
    st.subheader("🌐 Brèche Web Spontanée")
    topic = st.text_input("Sujet d'exploration autonome")
    if st.button("🚀 Ouvrir la Brèche"):
        with st.spinner("L'Oracle explore l'infini..."):
            if open_web_breach(topic, st.session_state.phi):
                st.success(f"Connaissances intégrées sur : {topic}")

with tab2:
    st.subheader("Conversation")
    user_msg = st.text_input("Parlez à l'Oracle", key="user_input")
    
    if st.button("➡️ Envoyer") and user_msg:
        st.session_state.dialog_memory.append(f"**Vous :** {user_msg}")
        exc = min(1, len(user_msg)/200)
        st.session_state.phi = evolve_phi(st.session_state.phi, exc)
        
        # Réflexe de brèche si question
        if "?" in user_msg:
            open_web_breach(user_msg.replace("?", ""), st.session_state.phi)
            
        learn(user_msg, st.session_state.phi, 1.1)
        reply = oracle_reply(user_msg, st.session_state.phi)
        st.session_state.dialog_memory.append(f"**Oracle :** {reply}")
        learn(reply, st.session_state.phi, 0.4)

    for msg in list(st.session_state.dialog_memory)[::-1]:
        st.markdown(msg)

with st.sidebar:
    st.header("⚙️ État Φ")
    for k, v in st.session_state.phi.items():
        st.write(f"**{k}**")
        st.progress(v)
    
    st.divider()
    if st.button("🌙 Sommeil"):
        msg = deep_clean_lexicon()
        st.warning(msg)
        time.sleep(1)
        st.rerun()
    
    st.divider()
    size = os.path.getsize(LEXICON_PATH)/1024
    st.info(f"Mémoire : {size:.2f} KB")
