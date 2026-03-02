# =====================================================
# 🧠 ORACLE V3 CONFORME — EXTENSION CORTICALE BIO-INSPIREE
# (V2 + couches corticales non destructives)
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
from collections import deque, Counter

# =====================================================
# 1. CONFIGURATION & STOCKAGE
# =====================================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")

os.makedirs(MEM_DIR, exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({}, open(LEXICON_PATH, "w", encoding="utf-8"))

# =====================================================
# 2. ÉTAT SESSION
# =====================================================
if "phi" not in st.session_state:
    st.session_state.phi = {
        "phi_m": 0.5,
        "phi_c": 0.5,
        "phi_d": 0.5
    }

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory = deque(maxlen=40)

# =====================================================
# 3. MOTEUR Φ (TRONC CÉRÉBRAL)
# =====================================================
def evolve_phi(phi, excitation):

    phi["phi_m"] = min(1, max(0.1, phi["phi_m"] + excitation*0.15 - 0.01))
    phi["phi_c"] = min(1, max(0.1, phi["phi_c"] + excitation*0.3 - 0.03))
    phi["phi_d"] = min(1, max(0.1, phi["phi_d"] + 0.02 - excitation*0.05))

    total = sum(phi.values())
    for key in phi:
        phi[key] /= total

    return phi

# =====================================================
# 4. MÉMOIRE
# =====================================================
def load_lex():
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_lex(L):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

# =====================================================
# 5. APPRENTISSAGE (PLASTICITÉ SYNAPTIQUE)
# =====================================================
def learn(text, phi, importance=1.0):

    if not text:
        return

    words = text.lower().split()
    if len(words) < 2:
        return

    L = load_lex()
    energy = math.sqrt(sum(v*v for v in phi.values())) * importance

    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + energy

    save_lex(L)

# =====================================================
# 6. SOMMEIL (ÉLAGAGE + DÉCROISSANCE)
# =====================================================
def deep_clean_lexicon(threshold=1.5):

    L = load_lex()
    clean_L = {}

    ban = ["http","www","uni00a0",".pdf",".docx","____"]

    for w, con in L.items():

        if len(w) < 2 or len(w) > 30 or any(b in w for b in ban):
            continue

        new = {
            t: v*0.999   # décroissance synaptique biologique
            for t, v in con.items()
            if v >= threshold and not any(b in t for b in ban)
        }

        if new:
            clean_L[w] = new

    save_lex(clean_L)
    return f"Sommeil terminé — {len(L)-len(clean_L)} connexions oubliées."

# =====================================================
# 7. SEED CONTEXTUEL (THALAMUS)
# =====================================================
def contextual_seed(L):

    context = " ".join(st.session_state.dialog_memory).split()
    candidates = [w for w in context if w in L]

    if candidates:
        return Counter(candidates).most_common(1)[0][0]

    return random.choice(list(L.keys()))

# =====================================================
# 🧠 8. CORTEX V3 — COUCHES SUPÉRIEURES
# =====================================================

def logical_layer(sequence, L):
    return [w for w in sequence if w in L] or sequence


def associative_layer(word, L, phi):

    if word not in L:
        return word

    options = L[word]

    if random.random() < phi["phi_c"]:
        return random.choices(
            list(options.keys()),
            weights=list(options.values())
        )[0]
    else:
        return max(options, key=options.get)


def predictive_layer(words, phi):

    if phi["phi_m"] > 0.6:
        words.append("continuité")

    if phi["phi_c"] > 0.65:
        words.append("évolution")

    return words

# =====================================================
# 9. GÉNÉRATION CORTICALE V3
# =====================================================
def oracle_reply(phi):

    L = load_lex()
    if not L:
        return "Mémoire vide. Nourrissez-moi."

    seed = contextual_seed(L)

    words = [seed]
    used = set(words)

    length = int(8 + phi["phi_m"]*35)

    for _ in range(length):

        filtered = logical_layer(words, L)
        current = filtered[-1]

        nxt = associative_layer(current, L, phi)

        if nxt in used and random.random() < phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    words = predictive_layer(words, phi)

    return " ".join(words).capitalize() + "."

# =====================================================
# 10. EXTRACTION MULTIMODALE (SNP)
# =====================================================
def extract_content(mode):

    raw_content = ""

    if mode == "Texte":
        raw_content = st.text_area("✍️ Entrez un texte pour nourrir l'Oracle")

    elif mode == "Document":
        file = st.file_uploader("📄 Charger PDF / Word / TXT",
                                type=["pdf","docx","txt"])
        if file:
            if file.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file)
                raw_content = " ".join(
                    p.extract_text() for p in reader.pages if p.extract_text()
                )
            elif file.name.endswith(".docx"):
                doc = docx.Document(file)
                raw_content = "\n".join(p.text for p in doc.paragraphs)
            else:
                raw_content = file.read().decode("utf-8")

    elif mode == "Excel":
        file = st.file_uploader("📊 Charger Excel", type=["xlsx","xls"])
        if file:
            df = pd.read_excel(file)
            raw_content = df.to_string()

    elif mode == "Audio":
        audio = st.file_uploader("🎙 Charger audio WAV", type="wav")
        if audio:
            r = sr.Recognizer()
            with sr.AudioFile(audio) as source:
                audio_data = r.record(source)
                try:
                    raw_content = r.recognize_google(audio_data, language="fr-FR")
                    st.info("Transcription réussie.")
                except:
                    st.error("Transcription impossible.")

    return raw_content

# =====================================================
# 11. INTERFACE
# =====================================================
st.set_page_config(page_title="ORACLE V3", page_icon="🧠", layout="wide")

st.title("🧠 ORACLE V3 — Cortex Cognitif Bio-Inspiré")

tab1, tab2 = st.tabs(["🌱 Nourrir l'Oracle", "💬 Parler à l'Oracle"])

# ---------- APPRENTISSAGE ----------
with tab1:

    st.subheader("Apprentissage Multimodal")

    mode = st.radio("Source du savoir",
                    ["Texte","Document","Excel","Audio"])

    content = extract_content(mode)

    if st.button("🌱 Nourrir l'Oracle"):
        if content:
            excitation = min(1, len(content)/500)
            st.session_state.phi = evolve_phi(
                st.session_state.phi, excitation
            )
            learn(content, st.session_state.phi, 1.3)
            st.success("L'Oracle a appris.")
        else:
            st.warning("Aucun contenu détecté.")

# ---------- CONVERSATION ----------
with tab2:

    st.subheader("Conversation")

    user_msg = st.text_input("Parlez à l'Oracle")

    colA, colB = st.columns([6,1])

    with colB:
        send = st.button("➡️")

    if send and user_msg:

        st.session_state.dialog_memory.append(user_msg)

        excitation = min(1, len(user_msg)/200)
        st.session_state.phi = evolve_phi(
            st.session_state.phi, excitation
        )

        learn(user_msg, st.session_state.phi, 1.1)

        reply = oracle_reply(st.session_state.phi)

        st.session_state.dialog_memory.append(reply)

        learn(reply, {"phi_m":0.1,"phi_c":0.1,"phi_d":0.1},0.3)

    for msg in st.session_state.dialog_memory:
        st.write(msg)

# =====================================================
# SIDEBAR ÉTAT
# =====================================================
with st.sidebar:

    st.header("🛠 État Cognitif")

    size = os.path.getsize(LEXICON_PATH)/1024
    st.success(f"Mémoire : {size:.2f} KB")

    st.divider()
    st.subheader("Φ Dynamique")

    for k,v in st.session_state.phi.items():
        st.progress(v, text=f"{k} : {v:.2f}")

    st.divider()
    if st.button("🌙 Sommeil (Nettoyage)"):
        msg = deep_clean_lexicon()
        st.warning(msg)
        time.sleep(1)
        st.rerun()

# =====================================================
# SAUVEGARDE
# =====================================================
st.divider()
st.subheader("💾 Mémoire Globale")

L = load_lex()

if L:
    c1,c2 = st.columns(2)

    with c1:
        st.download_button(
            "Télécharger mémoire",
            data=json.dumps(L,indent=2,ensure_ascii=False),
            file_name="lexicon.json"
        )

    with c2:
        restore = st.file_uploader("Restaurer mémoire", type="json")
        if restore:
            save_lex(json.load(restore))
            st.success("Mémoire restaurée.")
            st.rerun()