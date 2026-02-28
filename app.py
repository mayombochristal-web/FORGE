# =====================================================
# 🧠 ORACLE S+ — ARCHITECTURE COGNITIVE STABLE
# Pipeline officiel :
# S+01 → S+16
# =====================================================

# =====================================================
# S+01 — IMPORT_SYSTEM_CORE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time
import xml.etree.ElementTree as ET
from collections import Counter
from spectral_module import spectral_ui

# =====================================================
# S+02 — STREAMLIT_PAGE_CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE S+", layout="wide")

# =====================================================
# S+03 — MEMORY_PATH_MANAGER
# =====================================================

MEM = "oracle_memory"
os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

# =====================================================
# S+04 — SAFE_IO_LAYER
# =====================================================

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_frag():
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.to_csv(FILES["fragments"], index=False)

# =====================================================
# S+05 — MEMORY_INITIALIZER
# =====================================================

def init_memory():

    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(
            FILES["fragments"], index=False
        )

    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})

    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS":12,
            "age":0,
            "new_today":0,
            "last_day":str(datetime.date.today()),
            "timeline":[]
        })

init_memory()

# ==========================================================
# B2.5 — Cognitive Runtime Guard
# Stabilisation session + cohérence TTU runtime
# ==========================================================

import time
import uuid
import streamlit as st


# ----------------------------------------------------------
# Runtime ID unique (évite duplication session)
# ----------------------------------------------------------
def runtime_id():
    if "runtime_id" not in st.session_state:
        st.session_state.runtime_id = str(uuid.uuid4())
    return st.session_state.runtime_id


# ----------------------------------------------------------
# Horloge cognitive (∆t interne)
# ----------------------------------------------------------
def cognitive_clock():
    now = time.time()

    if "cognitive_time" not in st.session_state:
        st.session_state.cognitive_time = now
        st.session_state.delta_t = 0.0
    else:
        st.session_state.delta_t = now - st.session_state.cognitive_time
        st.session_state.cognitive_time = now

    return st.session_state.delta_t


# ----------------------------------------------------------
# Guard anti double-exécution Streamlit
# ----------------------------------------------------------
def runtime_guard():

    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.boot_time = time.time()
        st.session_state.rerun_count = 0
    else:
        st.session_state.rerun_count += 1


# ----------------------------------------------------------
# Synchronisation mémoire dialogue
# (corrige message invisible)
# ----------------------------------------------------------
def sync_dialog_memory():

    if "dialog" not in st.session_state:
        st.session_state.dialog = []

    if "pending_output" in st.session_state:
        st.session_state.dialog.append(
            st.session_state.pending_output
        )
        del st.session_state.pending_output


# ----------------------------------------------------------
# Bootstrap global
# ----------------------------------------------------------
def cognitive_bootstrap():

    runtime_guard()
    runtime_id()
    cognitive_clock()
    sync_dialog_memory()


# lancement automatique
cognitive_bootstrap()

# =====================================================
# S+06 — SHADOW_STATE_LOADER
# =====================================================

def sync_shadow():

    if "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])

        st.session_state.shadow_loaded = True

sync_shadow()

# =====================================================
# S+07 — COGNITIVE_TIME_TRACKER
# =====================================================

def cognitive_tick():

    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()

    st.session_state.cognitive_time = time.time() - st.session_state.t0

cognitive_tick()

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
# S+09 — LEARNING_ENGINE_CORE
# =====================================================

def learn(text):
    # --- Niveau caractère ---
    chars = char_tokens(text)

    # --- Niveau mot ---
    words = tokenize(text)

    if not words:
        return 0

    df = st.session_state.shadow_frag.copy()

    # Apprentissage fragments
    counts = Counter(words)

    for w,c in counts.items():
        mask = df["fragment"] == w
        if mask.any():
            df.loc[mask,"count"] += c
        else:
            df = pd.concat(
                [df, pd.DataFrame([[w,c]],columns=df.columns)],
                ignore_index=True
            )

    save_frag(df)
    st.session_state.shadow_frag = df

    # --- Relations TST ---
    assoc = st.session_state.shadow_rel

    # Relations caractères → caractères
    for i in range(len(chars)-1):
        a,b = chars[i],chars[i+1]
        assoc.setdefault(a,{})
        assoc[a][b] = assoc[a].get(b,0)+1

    # Relations mots → mots
    for i in range(len(words)-1):
        a,b = words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b] = assoc[a].get(b,0)+2

    save_json(FILES["relations"],assoc)

    # --- Cortex (Physique "Titan") ---
    cortex = st.session_state.shadow_cortex

    today = str(datetime.date.today())
    if cortex["last_day"] != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today

    # CALCUL DE L'ÂGE TITAN
    # 1 unité = 100 livres * 1000 pages * 2500 caractères
    UNITE_MASSIVE = 250_000_000 
    gain_age = len(chars) / UNITE_MASSIVE
    
    cortex["age"] += gain_age 
    cortex["new_today"] += len(counts)
    
    # VS ajusté pour rester sensible à l'échelle décimale
    cortex["VS"] = 10 + float(np.log1p(cortex["age"] * 1000))

    cortex["timeline"].extend(words[-50:])

    save_json(FILES["cortex"],cortex)

    return len(words)

# =====================================================
# S+10 — SEMANTIC_SEARCH_ENGINE
# =====================================================

def association_density():
    assoc=st.session_state.shadow_rel
    links=sum(len(v) for v in assoc.values())
    vocab=len(assoc)
    return round(links/max(vocab,1),2)

# =====================================================
# S+11 — PRETHINK_ENGINE
# =====================================================

def prethink(seed):

    assoc=st.session_state.shadow_rel

    if seed in assoc and assoc[seed]:
        return max(assoc[seed], key=assoc[seed].get)

    return seed
    
# =====================================================
# S+11B — LINGUISTIC OPERATOR L (TST)
# =====================================================

def linguistic_context(seed):

    assoc = st.session_state.shadow_rel

    if seed not in assoc:
        return {"context":"exploration"}

    neighbors = assoc[seed]

    themes = {}
    for w,score in neighbors.items():
        root = w[:4]
        themes[root] = themes.get(root,0)+score

    if not themes:
        return {"context":"vide"}

    context = max(themes, key=themes.get)

    return {
        "context":context,
        "strength":themes[context]
    }

# =====================================================
# S+12 — THINK_GENERATION_ENGINE
# =====================================================

def think(seed,steps=30):

    assoc=st.session_state.shadow_rel

    if seed not in assoc:
        return "Je dois encore apprendre."

    ctx = linguistic_context(seed)

    sent=[seed]
    cur=seed

    for _ in range(steps):

        nxt=assoc.get(cur)
        if not nxt:
            break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    sentence = " ".join(sent).capitalize()+"."

    return f"[Contexte:{ctx['context']}] {sentence}"

# =====================================================
# S+13 — COGNITIVE_METRICS
# =====================================================

def semantic_coherence():

    concepts=len(st.session_state.shadow_frag)
    assoc=len(st.session_state.shadow_rel)

    return round(min(100,(assoc/max(concepts,1))*10),2)

# =====================================================
# S+14 — AUTO_DIAGNOSTIC_SYSTEM
# =====================================================

def diagnose():
    cortex = st.session_state.shadow_cortex
    density = association_density()

    # Diagnostic basé sur l'activité récente et la densité du réseau
    if cortex["new_today"] < 10:
        return "🧠 Oracle en attente d'assimilation massive."

    if density < 2:
        return "🧠 Réseau en cours de structuration initiale."

    if density > 5:
        return "🧠 Sagesse Titan active : Résonance émergente détectée."

    return "🧠 Absorption de bibliothèques en cours."

# =====================================================
# S+15 — USER_DIALOG_INTERFACE
# =====================================================

st.title("🧠 ORACLE S+")

ctx=st.session_state.shadow_cortex

c1,c2,c3,c4=st.columns(4)
c1.metric("Vitalité",round(ctx["VS"],2))
c2.metric("Age",ctx["age"])
c3.metric("Densité",association_density())
c4.metric("Cohérence",semantic_coherence())

st.info(diagnose())

uploaded = st.file_uploader(
    "Nourrir l'IA",
    type=["txt","csv","docx","pdf"]
)

def read_docx(file):
    doc_bin = io.BytesIO(file.read())
    with zipfile.ZipFile(doc_bin) as z:
        xml = z.read("word/document.xml")
        tree = ET.fromstring(xml)
        texts = [
            node.text for node in tree.iter()
            if node.tag.endswith("t") and node.text
        ]
    return " ".join(texts)
    
if uploaded:
    # Lecture unique du fichier selon son extension
    if uploaded.name.endswith(".docx"):
        text = read_docx(uploaded)
    else:
        text = uploaded.getvalue().decode("utf-8","ignore")

    n = learn(text)
    st.success(f"{n} unités assimilées")

prompt = st.text_input("Intention")

if st.button("Penser"):
    tokens = tokenize(prompt)
    if tokens:
        seed = prethink(tokens[0])
        st.write(think(seed))
    else:
        st.warning("Phrase invalide.")

# =====================================================
# S+16 — STREAMLIT_UI_RENDER
# =====================================================

st.caption(
    f"Temps cognitif : {round(st.session_state.cognitive_time,2)} s"
)
spectral_ui(
    st.session_state.shadow_cortex,
    st.session_state.shadow_frag["fragment"].tolist()
)
