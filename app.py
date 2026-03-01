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
    if not text: return
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
    ban_list = ["uni00a0", "http", "www", "....", "____", "........", ".pdf", ".docx"]
    for word, connections in L.items():
        if any(b in word for b in ban_list) or len(word) > 30 or len(word) < 2:
            continue
        new_connections = {t: w for t, w in connections.items() if w >= threshold and not any(b in t for b in ban_list)}
        if new_connections:
            clean_L[word] = new_connections
    save_lex(clean_L)
    return f"Sommeil terminé. {len(L) - len(clean_L)} impuretés éliminées."

def oracle_reply(phi, seed=None):
    L = load_lex()
    if not L: return "Mémoire vide. Injectez un signal."
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
    words = [seed]
    # Longueur basée sur la Masse Phi_M
    for _ in range(int(5 + phi["phi_m"] * 25)):
        current = words[-1]
        if current not in L: break
        options = L[current]
        # Choix entre habitude (max) et créativité (random)
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

st.title("🧠 ORACLE : Système Cognitif Autonome")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🛠 État du Système")
    if os.path.exists(LEXICON_PATH):
        taille = os.path.getsize(LEXICON_PATH) / 1024
        st.success(f"Taille Mémoire : {taille:.2f} KB")
    
    st.divider()
    st.subheader("📊 Dynamique Φ")
    st.progress(st.session_state.phi["phi_m"], text=f"Masse (Mémoire) : {st.session_state.phi['phi_m']:.2f}")
    st.progress(st.session_state.phi["phi_c"], text=f"Chaleur (Créativité) : {st.session_state.phi['phi_c']:.2f}")
    st.progress(st.session_state.phi["phi_d"], text=f"Damping (Stabilité) : {st.session_state.phi['phi_d']:.2f}")

    st.divider()
    st.subheader("🌙 Maintenance (Sommeil)")
    seuil_val = st.slider("Seuil de survie synaptique", 1.0, 5.0, 1.5)
    if st.button("Lancer Deep Cleaning"):
        with st.spinner("Purification du lexique..."):
            msg = deep_clean_lexicon(threshold=seuil_val)
            st.warning(msg)
            time.sleep(1.5)
            st.rerun()

# --- CORPS DE L'APPLICATION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion Multimodale")
    mode = st.radio("Type de source :", ["Texte", "Document (PDF/Word/TXT)", "Excel", "Audio (WAV)"])
    raw_content = ""

    try:
        if mode == "Texte":
            raw_content = st.text_area("Saisissez le texte :", height=200)
        
        elif mode == "Document (PDF/Word/TXT)":
            uploaded_file = st.file_uploader("Charger Doc", type=["pdf", "docx", "txt"])
            if uploaded_file:
                if uploaded_file.name.endswith(".pdf"):
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    raw_content = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                elif uploaded_file.name.endswith(".docx"):
                    doc = docx.Document(uploaded_file)
                    raw_content = "\n".join([p.text for p in doc.paragraphs])
                else: # TXT
                    raw_content = uploaded_file.read().decode("utf-8")

        elif mode == "Excel":
            uploaded_file = st.file_uploader("Charger Tableur", type=["xlsx", "xls"])
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                raw_content = df.to_string()

        elif mode == "Audio (WAV)":
            audio_file = st.file_uploader("Charger Audio", type="wav")
            if audio_file:
                r = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio_data = r.record(source)
                    try:
                        raw_content = r.recognize_google(audio_data, language="fr-FR")
                        st.info(f"Transcription : {raw_content[:100]}...")
                    except:
                        st.error("L'API Google n'a pas pu transcrire l'audio.")
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")

    if st.button("⚡ Exciter l'Oracle") and raw_content:
        st.session_state.phi = evolve_phi(st.session_state.phi, 0.4)
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Signal intégré avec succès !")

with col2:
    st.subheader("💬 Réponse de l'Oracle")
    if st.button("Générer une Pensée"):
        with st.spinner("Ruminations internes..."):
            res = oracle_reply(st.session_state.phi)
            st.markdown(f"### > {res}")
            
            # Auto-apprentissage (Rumination)
            if random.random() < 0.3:
                learn_with_identity(res, {"phi_m":0.1, "phi_c":0.1, "phi_d":0.1})
                st.caption("✨ L'Oracle a auto-appris de cette phrase.")

# --- SAUVEGARDE ET RESTAURATION ---
st.divider()
st.subheader("💾 Mémoire Globale")
L_curr = load_lex()
if L_curr:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Télécharger le Lexicon (.json)", 
            data=json.dumps(L_curr, indent=2, ensure_ascii=False), 
            file_name="lexicon.json",
            mime="application/json"
        )
    with c2:
        restore = st.file_uploader("📂 Restaurer une mémoire externe", type="json")
        if restore:
            save_lex(json.load(restore))
            st.success("Mémoire synchronisée.")
            st.rerun()
