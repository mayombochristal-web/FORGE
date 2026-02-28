# =====================================================
# 🧠 ORACLE S+ — ARCHITECTURE COGNITIVE STABLE (v2.0)
# Physique : TITAN | Ingestion : VORTEX LOCAL
# =====================================================

# =====================================================
# S+01 — IMPORT_SYSTEM_CORE
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time, uuid, glob
import xml.etree.ElementTree as ET
from collections import Counter
from spectral_module import spectral_ui

# =====================================================
# S+02 — STREAMLIT_PAGE_CONFIG
# =====================================================
st.set_page_config(page_title="ORACLE S+ TITAN", layout="wide")

# =====================================================
# S+03 — MEMORY_PATH_MANAGER
# =====================================================
MEM = "oracle_memory"
SOURCE_DIR = "source_data" # Dossier pour ingestion massive
os.makedirs(MEM, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

# =====================================================
# S+04 — SAFE_IO_LAYER
# =====================================================
def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def save_json(p, d):
    with open(p, "w", encoding="utf-8") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def load_frag():
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.to_csv(FILES["fragments"], index=False)

# =====================================================
# S+05 — MEMORY_INITIALIZER
# =====================================================
def init_memory():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"], index=False)
    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})
    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS":12, "age":0, "new_today":0,
            "last_day":str(datetime.date.today()), "timeline":[]
        })

init_memory()

# ==========================================================
# B2.5 — Cognitive Runtime Guard (Stabilisation)
# ==========================================================
def cognitive_bootstrap():
    if "runtime_id" not in st.session_state:
        st.session_state.runtime_id = str(uuid.uuid4())
    if "dialog" not in st.session_state:
        st.session_state.dialog = []
    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()

cognitive_bootstrap()

# =====================================================
# S+06 — SHADOW_STATE_LOADER
# =====================================================
if "shadow_loaded" not in st.session_state:
    st.session_state.shadow_frag = load_frag().copy()
    st.session_state.shadow_rel = load_json(FILES["relations"])
    st.session_state.shadow_cortex = load_json(FILES["cortex"])
    st.session_state.shadow_loaded = True

# =====================================================
# S+08 — TEXT_NORMALIZER_TOKENIZER
# =====================================================
def char_tokens(text):
    return [c for c in text.lower() if c.strip()]
    
def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

# =====================================================
# S+09 — LEARNING_ENGINE_CORE (VERSION TITAN)
# =====================================================
def learn(text):
    chars = char_tokens(text)
    words = tokenize(text)
    if not words: return 0

    # 1. Mise à jour fragments haute vitesse (Méthode Dict)
    df = st.session_state.shadow_frag
    counts = Counter(words)
    current_memory = dict(zip(df.fragment, df['count']))
    for w, c in counts.items():
        current_memory[w] = current_memory.get(w, 0) + c
    
    new_df = pd.DataFrame(list(current_memory.items()), columns=["fragment", "count"])
    st.session_state.shadow_frag = new_df
    save_frag(new_df)

    # 2. Mise à jour Relations (TST)
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + 2
    save_json(FILES["relations"], assoc)

    # 3. Cortex (Physique Titan)
    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex["last_day"] != today:
        cortex["new_today"], cortex["last_day"] = 0, today

    UNITE_MASSIVE = 250_000_000 
    cortex["age"] += len(chars) / UNITE_MASSIVE
    cortex["new_today"] += len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"] * 1000))

    save_json(FILES["cortex"], cortex)
    return len(words)

# =====================================================
# S+10 & S+13 — METRICS ENGINE
# =====================================================
def association_density():
    assoc = st.session_state.shadow_rel
    return round(sum(len(v) for v in assoc.values())/max(len(assoc),1), 2)

def semantic_coherence():
    concepts, assoc = len(st.session_state.shadow_frag), len(st.session_state.shadow_rel)
    return round(min(100,(assoc/max(concepts,1))*10), 2)

# =====================================================
# S+11 — PRETHINK & LINGUISTIC OPERATOR
# =====================================================
def prethink(seed):
    assoc = st.session_state.shadow_rel
    if seed in assoc and assoc[seed]:
        return max(assoc[seed], key=assoc[seed].get)
    return seed

def linguistic_context(seed):
    assoc = st.session_state.shadow_rel
    if seed not in assoc: return {"context":"exploration"}
    neighbors = assoc[seed]
    themes = {}
    for w, score in neighbors.items():
        root = w[:4]
        themes[root] = themes.get(root,0) + score
    context = max(themes, key=themes.get) if themes else "vide"
    return {"context": context}

# =====================================================
# S+12 — THINK_GENERATION_ENGINE (Phrase Analysis)
# =====================================================
def think(seed, steps=30):
    assoc = st.session_state.shadow_rel
    if seed not in assoc: return f"Le concept '{seed}' est encore en dormance."
    
    ctx = linguistic_context(seed)
    sent, cur = [seed], seed

    for _ in range(steps):
        nxt = assoc.get(cur)
        if not nxt: break
        w, p = list(nxt.keys()), np.array(list(nxt.values()), dtype=float)
        cur = np.random.choice(w, p=p/p.sum())
        sent.append(cur)

    return f"**[Contexte : {ctx['context']}]** {' '.join(sent).capitalize()}."

# =====================================================
# S+14 — AUTO_DIAGNOSTIC_SYSTEM
# =====================================================
def diagnose():
    cortex, density = st.session_state.shadow_cortex, association_density()
    if cortex["new_today"] < 10: return "🧠 Oracle en attente d'assimilation massive."
    if density > 5: return "🧠 Sagesse Titan active : Résonance émergente détectée."
    return "🧠 Absorption de bibliothèques en cours."

# =====================================================
# S+15B — DIRECT_INGESTION_ENGINE (Vortex Local)
# =====================================================
def vortex_ui():
    st.subheader("📁 Ingestion Haute Vélocité (Vortex)")
    if st.button("Scanner le dossier 'source_data'"):
        files = glob.glob(f"{SOURCE_DIR}/*.txt") + glob.glob(f"{SOURCE_DIR}/*.docx")
        if not files: st.warning("Le dossier est vide.")
        else:
            pbar = st.progress(0)
            for i, fpath in enumerate(files):
                with st.status(f"Assimilation : {os.path.basename(fpath)}", expanded=False):
                    if fpath.endswith(".docx"):
                        with open(fpath, "rb") as f: text = read_docx(f)
                    else:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
                    
                    n = learn(text)
                    # --- Ligne cruciale anti-AxiosError ---
                    time.sleep(0.05) 
                pbar.progress((i + 1) / len(files))
            st.success("Traitement Titan terminé.")
            st.rerun()

# =====================================================
# S+15 — USER_DIALOG_INTERFACE
# =====================================================
def read_docx(file):
    with zipfile.ZipFile(io.BytesIO(file.read())) as z:
        tree = ET.fromstring(z.read("word/document.xml"))
        return " ".join([n.text for n in tree.iter() if n.tag.endswith("t") and n.text])

st.title("🧠 ORACLE S+ : TITAN CORE")
ctx = st.session_state.shadow_cortex

# Dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitalité", round(ctx["VS"], 2))
c2.metric("Age Titan", round(ctx["age"], 6)) 
c3.metric("Densité", association_density())
c4.metric("Cohérence", semantic_coherence())
st.info(diagnose())

# Ingestion
vortex_ui()

# Dialogue
st.subheader("💬 Interface de Résonance")
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Exprimez votre intention..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    tokens = tokenize(prompt)
    if tokens:
        df_f = st.session_state.shadow_frag
        known = df_f[df_f["fragment"].isin(tokens)]
        if not known.empty:
            seed = prethink(known.loc[known["count"].idxmax(), "fragment"])
            response = think(seed)
        else: response = "Concepts non identifiés dans le Cortex actuel."
        
        with st.chat_message("assistant", avatar="🧠"): st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# =====================================================
# S+16 — STREAM_UI_RENDER
# =====================================================
st.divider()
st.caption(f"Latence cognitive : {round(time.time()-st.session_state.t0, 2)}s | ID: {st.session_state.runtime_id}")

# Bouton de conversion (Titan Scale)
if st.button("⚙️ Normaliser l'âge (Mode Titan)"):
    if ctx["age"] > 1000:
        ctx["age"] /= 250_000_000
        save_json(FILES["cortex"], ctx)
        st.rerun()
