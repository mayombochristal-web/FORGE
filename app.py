# =====================================================
# 🧠 ORACLE S+ — ARCHITECTURE COGNITIVE STABLE (v2)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, datetime, time, uuid, glob
from collections import Counter

# <<< ajout pour le module mémoire >>>
from oracle_memory.oracle_memory import (
    init_memory,
    load_frag,
    save_frag,
    load_json,
    save_json,
    merge_fragments,
    merge_relations,
    merge_cortex,
    FILES,   # si tu exposes ce dict dans oracle_memory.py
)

# Spectral UI (safe import)
try:
    from spectral_module import spectral_ui
except ImportError:
    def spectral_ui(*args, **kwargs):
        pass

# DOCX support
try:
    from docx import Document
except ImportError:
    Document = None

# PDF support
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(page_title="ORACLE S+", layout="wide")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
MEM_DIR = os.path.join(BASE_DIR, "oracle_memory")
os.makedirs(MEM_DIR, exist_ok=True)

FILES = {
    "fragments": os.path.join(MEM_DIR, "fragments.csv"),
    "relations": os.path.join(MEM_DIR, "relations.json"),
    "cortex":    os.path.join(MEM_DIR, "cortex.json"),
}

# =====================================================
# UTILITAIRES I/O
# =====================================================

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_frag():
    return pd.read_csv(FILES["fragments"]) if os.path.exists(FILES["fragments"]) else pd.DataFrame(columns=["fragment", "count"])

def save_frag(df):
    df.to_csv(FILES["fragments"], index=False)

# =====================================================
# INITIALISATION DE LA MÉMOIRE
# =====================================================

def init_memory():
    if not os.path.exists(FILES["fragments"]):
        save_frag(pd.DataFrame(columns=["fragment", "count"]))
    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})
    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS": 12, "age": 0, "new_today": 0,
            "last_day": str(datetime.date.today()), "timeline": []
        })

init_memory()

# =====================================================
# SHADOW STATE MANAGER
# =====================================================

def sync_shadow(force=False):
    """Recharge la mémoire complète dans la session Streamlit."""
    if force or "shadow_loaded" not in st.session_state:
        st.session_state.shadow_frag = load_frag()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])
        st.session_state.shadow_loaded = True

def safe_reload(): sync_shadow(force=True)
sync_shadow()

# =====================================================
# FONCTIONS TEXTE
# =====================================================

def clean(text):
    return re.sub(r"[^wàâéèêëîïôùûüœs]", " ", text.lower())

def tokenize(text):
    return [w for w in clean(text).split() if len(w) > 2 and not w.isnumeric()]

# =====================================================
# LECTEURS DE FICHIER
# =====================================================

def read_docx(file):
    if Document is None: return ""
    return "
".join(p.text for p in Document(file).paragraphs)

def read_pdf(file):
    if PdfReader is None: return ""
    try:
        reader = PdfReader(file)
        text_pages = [p.extract_text() or "" for p in reader.pages]
        text = "
".join(text_pages)
        # filtre PDF corrompu
        if re.search(r"obj|stream|flatedecode", text, re.IGNORECASE):
            st.warning("⚠️ Contenu PDF non textuel détecté — filtrage effectué.")
            text = re.sub(r"\b(obj|endobj|stream|flatedecode)\b", "", text)
        return text
    except Exception:
        return ""

# =====================================================
# APPRENTISSAGE
# =====================================================

def learn(text):
    if not text.strip():
        return 0

    words = tokenize(text)
    chars = [c for c in text if c.strip()]
    if not words:
        return 0

    df = st.session_state.shadow_frag
    fragments = dict(zip(df["fragment"], df["count"]))
    counts = Counter(words)

    for w, c in counts.items():
        fragments[w] = fragments.get(w, 0) + c

    new_df = pd.DataFrame(list(fragments.items()), columns=["fragment", "count"])
    st.session_state.shadow_frag = new_df
    save_frag(new_df)

    assoc = st.session_state.shadow_rel
    window = 3  # fenêtre de lien élargie
    for i in range(len(words)):
        a = words[i]
        assoc.setdefault(a, {})
        for j in range(1, window + 1):
            if i + j < len(words):
                b = words[i + j]
                assoc[a][b] = assoc[a].get(b, 0) + (window - j + 1)
    save_json(FILES["relations"], assoc)

    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex["last_day"] != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today

    UNITE_MASSIVE = 250_000_000
    cortex["age"] += len(chars) / UNITE_MASSIVE
    cortex["new_today"] += len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"] * 1000))
    save_json(FILES["cortex"], cortex)

    safe_reload()
    return len(words)

# =====================================================
# MOTEUR COGNITIF
# =====================================================

def prethink(seed):
    assoc = st.session_state.shadow_rel
    if seed in assoc and assoc[seed]:
        return max(assoc[seed], key=assoc[seed].get)
    return seed

def think(seed, steps=30, temp=1.0):
    assoc = st.session_state.shadow_rel
    if seed not in assoc or not assoc[seed]:
        return f"Le concept '{seed}' est isolé dans ma structure."

    sent, cur = [seed], seed
    for _ in range(steps):
        nxt = assoc.get(cur)
        if not nxt: break
        words = list(nxt.keys())
        probs = np.array(list(nxt.values()), dtype=float)
        probs **= 1 / temp
        probs /= probs.sum()
        cur = np.random.choice(words, p=probs)
        sent.append(cur)
    return " ".join(sent).capitalize() + "."

# =====================================================
# MÉTRIQUES SÉMANTIQUES
# =====================================================

def association_density():
    assoc = st.session_state.shadow_rel
    vocab = len(assoc)
    links = sum(len(v) for v in assoc.values())
    return round(links / max(vocab, 1), 2)

def delta_k_runtime():
    ctx = st.session_state.shadow_cortex
    ratio = ctx.get("new_today", 0) / max(ctx.get("age", 1), 1)
    ctx["delta_ratio"] = ratio
    if ratio > 0.30:
        st.warning("⚠️ Surcharge cognitive détectée (∆k/k élevé)")

# =====================================================
# INTERFACE PRINCIPALE
# =====================================================

st.title("🧠 ORACLE S+")

ctx = st.session_state.shadow_cortex
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitalité", round(ctx["VS"], 2))
c2.metric("Âge", round(ctx["age"], 6))
c3.metric("Densité", association_density())
c4.metric("Concepts", len(st.session_state.shadow_frag))
delta_k_runtime()

# Chat interactif
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Entrez un concept..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    tokens = tokenize(prompt)
    response = "Flux insuffisant."
    if tokens:
        seed = prethink(tokens[0])
        response = think(seed)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# =====================================================
# PANNEAU D'IMPORT / EXPORT
# =====================================================

st.divider()
st.subheader("💾 Mémoire Oracle — Import / Export")

col1, col2, col3 = st.columns(3)
with col1:
    if os.path.exists(FILES["fragments"]):
        with open(FILES["fragments"], "rb") as f:
            st.download_button("⬇️ fragments.csv", f, "oracle_fragments.csv", "text/csv", use_container_width=True)
with col2:
    if os.path.exists(FILES["relations"]):
        with open(FILES["relations"], "rb") as f:
            st.download_button("⬇️ relations.json", f, "oracle_relations.json", "application/json", use_container_width=True)
with col3:
    if os.path.exists(FILES["cortex"]):
        with open(FILES["cortex"], "rb") as f:
            st.download_button("⬇️ cortex.json", f, "oracle_cortex.json", "application/json", use_container_width=True)

uploaded = st.file_uploader("Importer une mémoire ou un texte (.csv, .json, .zip, .pdf, .docx, .txt)")

if uploaded:
    try:
        # Texte bruts
        if uploaded.name.endswith(".docx"):
            text = read_docx(uploaded)
            n = learn(text)
            st.success(f"✅ {n} mots appris.")
        elif uploaded.name.endswith(".pdf"):
            text = read_pdf(uploaded)
            n = learn(text)
            st.success(f"✅ {n} mots appris.")
        elif uploaded.name.endswith(".txt"):
            text = uploaded.read().decode("utf-8", errors="ignore")
            n = learn(text)
            st.success(f"✅ {n} mots appris.")
        # Mémoire
        elif uploaded.name.endswith(".csv"):
            incoming_df = pd.read_csv(uploaded)
            merged_df = pd.concat([load_frag(), incoming_df]).groupby("fragment", as_index=False).sum()
            save_frag(merged_df); safe_reload()
            st.success("✅ fragments fusionnés.")
        elif uploaded.name.endswith(".json"):
            data = json.load(uploaded)
            if "timeline" in data:
                ctx = merge_cortex(load_json(FILES["cortex"]), data)
                save_json(FILES["cortex"], ctx)
                st.success("✅ cortex fusionné.")
            else:
                rel = merge_relations(load_json(FILES["relations"]), data)
                save_json(FILES["relations"], rel)
                st.success("✅ relations fusionnées.")
            safe_reload()
        elif uploaded.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded.read())) as z:
                for name in z.namelist():
                    if name.endswith("fragments.csv"):
                        df = pd.read_csv(z.open(name))
                        merged_df = pd.concat([load_frag(), df]).groupby("fragment", as_index=False).sum()
                        save_frag(merged_df)
                    elif name.endswith("relations.json"):
                        rel = merge_relations(load_json(FILES["relations"]), json.load(z.open(name)))
                        save_json(FILES["relations"], rel)
                    elif name.endswith("cortex.json"):
                        ctx = merge_cortex(load_json(FILES["cortex"]), json.load(z.open(name)))
                        save_json(FILES["cortex"], ctx)
            safe_reload()
            st.success("✅ Mémoire ZIP fusionnée.")
    except Exception as e:
        st.error(f"❌ Import impossible : {e}")

# =====================================================
# VISUALISATION SPECTRALE
# =====================================================

st.caption(f"Temps cognitif : {round(time.time(),2)} s")
spectral_ui(st.session_state.shadow_cortex, st.session_state.shadow_frag["fragment"].tolist())

# =====================================================
# OUTILS DE FUSION (RÉUTILISÉS POUR ZIP/JSON)
# =====================================================

def merge_fragments(local_df, incoming_df):
    return pd.concat([local_df, incoming_df]).groupby("fragment", as_index=False).sum()

def merge_relations(local_rel, incoming_rel):
    for a, links in incoming_rel.items():
        local_rel.setdefault(a, {})
        for b, w in links.items():
            local_rel[a][b] = local_rel[a].get(b, 0) + w
    return local_rel

def merge_cortex(local_ctx, incoming_ctx):
    local_ctx["age"] += incoming_ctx.get("age", 0)
    local_ctx["new_today"] += incoming_ctx.get("new_today", 0)
    local_ctx["timeline"].extend(incoming_ctx.get("timeline", []))
    local_ctx["timeline"] = local_ctx["timeline"][-5000:]
    local_ctx["VS"] = 10 + float(np.log1p(local_ctx["age"] * 1000))
    return local_ctx