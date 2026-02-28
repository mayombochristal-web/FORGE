# =====================================================
# 🧠 ORACLE V12 — RÉSONANCE & MULTI-FORMAT
# TST Ghost Memory + Multi-Source Ingestion + Spectral Resonance
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

# Modules scientifiques pour l'analyse spectrale
try:
    from scipy.signal import welch, stft
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

# Configuration de la page
st.set_page_config(page_title="ORACLE V12 — SYSTÈME GLOBAL", layout="wide")

# --------------------------------------------------
# CONFIGURATION DE L'ANCRAGE (MÉMOIRE PERMANENTE)
# --------------------------------------------------
MEM = "oracle_memory"
os.makedirs(MEM, exist_ok=True)
FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "concepts": f"{MEM}/concepts.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

def init_memory():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment", "count"]).to_csv(FILES["fragments"], index=False)
    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept", "weight"]).to_csv(FILES["concepts"], index=False)
    if not os.path.exists(FILES["relations"]):
        json.dump({}, open(FILES["relations"], "w"))
    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "VS": 12.0, 
            "age": 0, 
            "timeline": [], 
            "last_day": str(datetime.date.today()),
            "new_today": 0
        }, open(FILES["cortex"], "w"))

init_memory()

# --------------------------------------------------
# SYNC SHADOW STATE (FANTÔME)
# --------------------------------------------------
def sync_shadow():
    if "shadow_loaded" not in st.session_state:
        st.session_state.shadow_frag = pd.read_csv(FILES["fragments"])
        st.session_state.shadow_rel = json.load(open(FILES["relations"]))
        st.session_state.shadow_cortex = json.load(open(FILES["cortex"]))
        st.session_state.shadow_loaded = True

sync_shadow()

# --------------------------------------------------
# LECTURE MULTI-FORMATS (RÉINTÉGRÉE)
# --------------------------------------------------
def read_source_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            return file.read().decode("utf-8", "ignore")
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string()
        if name.endswith(".xlsx"):
            df = pd.read_excel(file)
            return df.to_string()
        if name.endswith(".docx"):
            doc = zipfile.ZipFile(io.BytesIO(file.read()))
            xml_content = doc.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            return " ".join(t.text for t in tree.iter() if t.text)
        if name.endswith(".pdf"):
            # Lecture brute type TST (analyse spectrale des flux)
            return file.read().decode("latin-1", "ignore")
    except Exception as e:
        st.error(f"Erreur de lecture sur {name}: {e}")
        return ""
    return ""

# --------------------------------------------------
# CORE LINGUISTIQUE (TST LEARNING)
# --------------------------------------------------
def clean_text(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean_text(t).split() if len(w) > 1]

def learn(text):
    words = tokenize(text)
    if not words: return 0

    # 1. Mise à jour Fragments (Fréquences)
    df = st.session_state.shadow_frag
    counts = Counter(words)
    for w, c in counts.items():
        mask = df["fragment"] == w
        if mask.any():
            df.loc[mask, "count"] += c
        else:
            df = pd.concat([df, pd.DataFrame([{"fragment": w, "count": c}])], ignore_index=True)
    df.to_csv(FILES["fragments"], index=False)
    st.session_state.shadow_frag = df

    # 2. Mise à jour Relations (Couplage sémantique)
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + 2
    json.dump(assoc, open(FILES["relations"], "w"))
    st.session_state.shadow_rel = assoc

    # 3. Mise à jour Cortex & Timeline
    cortex = st.session_state.shadow_cortex
    cortex["age"] += len(words)
    cortex["new_today"] += len(counts)
    cortex["timeline"].extend(words)
    # Protection RAM : on garde les 30k derniers mots pour l'analyse spectrale
    if len(cortex["timeline"]) > 30000:
        cortex["timeline"] = cortex["timeline"][-30000:]
    
    cortex["VS"] = 10 + float(np.log1p(cortex["age"]))
    json.dump(cortex, open(FILES["cortex"], "w"))
    st.session_state.shadow_cortex = cortex

    return len(words)

# --------------------------------------------------
# FONCTIONS SPECTRALES
# --------------------------------------------------
def get_psd(word, timeline, nperseg=128):
    signal = np.array([1.0 if w == word else 0.0 for w in timeline])
    if len(signal) < nperseg: return None, None
    f, pxx = welch(signal, fs=1.0, nperseg=nperseg)
    return f, pxx

# --------------------------------------------------
# INTERFACE UTILISATEUR
# --------------------------------------------------
st.title("🧠 ORACLE V12 — INTELLIGENCE SPECTRALE UNIFIÉE")

# Section des Métriques
ctx = st.session_state.shadow_cortex
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vitalité Spectrale", f"{round(ctx['VS'], 2)}")
m2.metric("Âge Cognitif", ctx["age"])
m3.metric("Mots Appris (24h)", ctx["new_today"])
m4.metric("Fragments en Base", len(st.session_state.shadow_frag))

# --- SIDEBAR : INGESTION ---
with st.sidebar:
    st.header("📥 Ingestion de Données")
    st.info("Supporte : PDF, DOCX, XLSX, CSV, TXT")
    uploaded_files = st.file_uploader("Nourrir l'IA", accept_multiple_files=True)
    
    if st.button("Lancer l'assimilation"):
        if uploaded_files:
            total_words = 0
            for f in uploaded_files:
                content = read_source_file(f)
                if content:
                    total_words += learn(content)
            st.success(f"Cortex mis à jour : +{total_words} mots.")
            st.rerun()

# --- ANALYSE SPECTRALE ---
st.subheader("🔬 Laboratoire de Résonance")
if len(ctx["timeline"]) > 150:
    word_list = st.session_state.shadow_frag.nlargest(100, "count")["fragment"].tolist()
    col_sel, col_graph = st.columns([1, 2])
    
    with col_sel:
        target = st.selectbox("Analyser la signature de :", word_list)
        nper = st.slider("Précision (nperseg)", 32, 512, 128, step=32)
    
    if SPECTRAL_AVAILABLE:
        f, pxx = get_psd(target, ctx["timeline"], nperseg=nper)
        if f is not None:
            with col_graph:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.semilogy(f, pxx, color="#00FFAA", lw=2)
                ax.set_title(f"Densité Spectrale de '{target}'")
                ax.set_xlabel("Fréquence (cycles/mot)")
                ax.grid(True, which="both", ls="-", alpha=0.2)
                st.pyplot(fig)
        else:
            st.warning("Données insuffisantes pour ce mot avec cette précision.")
else:
    st.info("Nourrissez l'IA avec plus de texte pour activer l'analyse de fréquence.")

# --- DIALOGUE COGNITIF ---
st.subheader("💬 Dialogue avec l'Oracle")
seed_input = st.text_input("Germe de pensée (ex: maintenance)")
if st.button("Penser"):
    tokens = tokenize(seed_input)
    if tokens and tokens[0] in st.session_state.shadow_rel:
        seed = tokens[0]
        sentence = [seed]
        current = seed
        for _ in range(25):
            next_options = st.session_state.shadow_rel.get(current, {})
            if not next_options: break
            choices = list(next_options.keys())
            weights = np.array(list(next_options.values()), dtype=float)
            current = np.random.choice(choices, p=weights/weights.sum())
            sentence.append(current)
        st.markdown(f"**L'Oracle dit :** *{' '.join(sentence).capitalize()}.*")
    else:
        st.write("Ce concept n'est pas encore ancré dans ma mémoire.")

# --- EXPORT & SÉCURITÉ ---
st.divider()
c_exp, c_del = st.columns(2)
with c_exp:
    if st.button("📦 Préparer Exportation du Cerveau"):
        full_data = {
            "fragments": st.session_state.shadow_frag.to_dict(orient="records"),
            "relations": st.session_state.shadow_rel,
            "cortex": st.session_state.shadow_cortex
        }
        st.download_button(
            "📥 Télécharger oracle_v12.json",
            json.dumps(full_data, indent=2, ensure_ascii=False),
            file_name=f"oracle_v12_{datetime.date.today()}.json"
        )
