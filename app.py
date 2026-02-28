# =====================================================
# 🧠 ORACLE V12 — RÉSONANCE & HARMONIQUES
# TST Ghost Memory + Analyse de Phase Groupée
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

# Modules scientifiques
try:
    from scipy.signal import stft, welch
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

# Configuration
st.set_page_config(page_title="ORACLE V12 RESONANCE", layout="wide")

# --------------------------------------------------
# SYSTÈME DE FICHIERS (ANCRAGE)
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
        json.dump({"VS": 12, "age": 0, "timeline": [], "last_day": str(datetime.date.today())}, open(FILES["cortex"], "w"))

init_memory()

# --------------------------------------------------
# SHADOW STATE (ÉTAT FANTÔME)
# --------------------------------------------------
def sync_shadow():
    if "shadow_loaded" not in st.session_state:
        try:
            st.session_state.shadow_frag = pd.read_csv(FILES["fragments"])
            st.session_state.shadow_rel = json.load(open(FILES["relations"]))
            st.session_state.shadow_cortex = json.load(open(FILES["cortex"]))
            st.session_state.shadow_loaded = True
        except:
            st.session_state.shadow_loaded = False

sync_shadow()

# --------------------------------------------------
# CORE LINGUISTIQUE (OPÉRATEUR L)
# --------------------------------------------------
def clean(t): return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())
def tokenize(t): return [w for w in clean(t).split() if len(w) > 1]

def learn(text):
    words = tokenize(text)
    if not words: return 0

    # Mise à jour fragments
    df = st.session_state.shadow_frag
    counts = Counter(words)
    new_rows = []
    for w, c in counts.items():
        if w in df["fragment"].values:
            df.loc[df["fragment"] == w, "count"] += c
        else:
            new_rows.append({"fragment": w, "count": c})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    df.to_csv(FILES["fragments"], index=False)
    st.session_state.shadow_frag = df

    # Mise à jour relations & cortex
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})[b] = assoc[a].get(b, 0) + 2
    
    json.dump(assoc, open(FILES["relations"], "w"))
    
    cortex = st.session_state.shadow_cortex
    cortex["age"] += len(words)
    cortex["timeline"].extend(words)
    # Limitation de la timeline pour éviter l'explosion RAM (Keep last 50k words)
    if len(cortex["timeline"]) > 50000: cortex["timeline"] = cortex["timeline"][-50000:]
    
    cortex["VS"] = 10 + float(np.log1p(cortex["age"]))
    json.dump(cortex, open(FILES["cortex"], "w"))
    
    return len(words)

# --------------------------------------------------
# ANALYSE SPECTRALE DE RÉSONANCE
# --------------------------------------------------
def get_spectral_signature(word, timeline, nperseg=128):
    signal = np.array([1.0 if w == word else 0.0 for w in timeline])
    if len(signal) < nperseg: return None
    f, pxx = welch(signal, fs=1.0, nperseg=nperseg)
    return f, pxx

def find_resonances(target_word, threshold=0.8):
    timeline = st.session_state.shadow_cortex["timeline"]
    f_target, p_target = get_spectral_signature(target_word, timeline)
    
    resonances = []
    # On teste les 50 mots les plus fréquents pour la résonance
    top_words = st.session_state.shadow_frag.nlargest(50, "count")["fragment"].tolist()
    
    for word in top_words:
        if word == target_word: continue
        sig = get_spectral_signature(word, timeline)
        if sig:
            f, p = sig
            correlation = np.corrcoef(p_target, p)[0, 1]
            if correlation > threshold:
                resonances.append((word, correlation))
    return sorted(resonances, key=lambda x: x[1], reverse=True)

# --------------------------------------------------
# INTERFACE
# --------------------------------------------------
st.title("🧠 ORACLE V12 — RÉSONANCE SPECTRALE")

# DASHBOARD VITALITÉ
c1, c2, c3 = st.columns(3)
c1.metric("Vitalité (VS)", round(st.session_state.shadow_cortex["VS"], 2))
c2.metric("Âge Cognitif", st.session_state.shadow_cortex["age"])
c3.metric("Fragments", len(st.session_state.shadow_frag))

# INPUT
with st.sidebar:
    st.header("📥 Ingestion")
    file = st.file_uploader("Nourrir l'Oracle", type=["txt", "pdf", "docx"])
    if file:
        # Lecture simplifiée pour l'exemple
        raw_text = file.read().decode("utf-8", "ignore")
        n = learn(raw_text)
        st.success(f"{n} unités intégrées.")

# ANALYSE DE RÉSONANCE
st.subheader("🔬 Analyse de Résonance (Couplage de Phase)")
if len(st.session_state.shadow_cortex["timeline"]) > 200:
    word_list = st.session_state.shadow_frag.nlargest(100, "count")["fragment"].tolist()
    selected_word = st.selectbox("Sélectionner un concept maître", word_list)
    
    if st.button("Calculer les Résonances"):
        if SPECTRAL_AVAILABLE:
            res = find_resonances(selected_word)
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("### Mots en résonance")
                if res:
                    for w, c in res[:10]:
                        st.write(f"**{w}** : {round(c*100,1)}% de couplage")
                else:
                    st.write("Aucune résonance forte détectée.")
            
            with col_b:
                # Graphique des densités spectrales
                timeline = st.session_state.shadow_cortex["timeline"]
                f, p_target = get_spectral_signature(selected_word, timeline)
                fig, ax = plt.subplots()
                ax.plot(f, p_target, label=selected_word, lw=2)
                if res:
                    f2, p2 = get_spectral_signature(res[0][0], timeline)
                    ax.plot(f2, p2, label=f"Résonance: {res[0][0]}", alpha=0.7)
                ax.set_title("Signature de Densité Spectrale (PSD)")
                ax.legend()
                st.pyplot(fig)
        else:
            st.error("Installez scipy/matplotlib pour cette fonction.")
else:
    st.info(" Timeline insuffisante pour l'analyse spectrale (minimum 200 mots requis).")

# PENSÉE
st.subheader("💬 Dialogue avec l'Oracle")
prompt = st.text_input("Saisir un germe de pensée")
if prompt:
    seed = tokenize(prompt)[0] if tokenize(prompt) else ""
    if seed in st.session_state.shadow_rel:
        sent = [seed]
        curr = seed
        for _ in range(20):
            nxts = st.session_state.shadow_rel.get(curr, {})
            if not nxts: break
            curr = np.random.choice(list(nxts.keys()), p=np.array(list(nxts.values()))/sum(nxts.values()))
            sent.append(curr)
        st.write(f"**Oracle :** {' '.join(sent).capitalize()}.")
    else:
        st.write("Concept inconnu de mon ancrage actuel.")

# EXPORT
st.divider()
if st.button("Exporter le Cerveau (JSON)"):
    data = {
        "fragments": st.session_state.shadow_frag.to_dict(),
        "relations": st.session_state.shadow_rel,
        "cortex": st.session_state.shadow_cortex
    }
    st.download_button("Télécharger", json.dumps(data), "oracle_v12_brain.json")
