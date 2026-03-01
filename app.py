# =====================================================
# 🧠 ORACLE S+ — ARCHITECTURE COGNITIVE STABLE
# =====================================================

# =====================================================
# S+01 — IMPORT_SYSTEM_CORE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, datetime, time, uuid, glob
from collections import Counter

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


# =====================================================
# S+02 — STREAMLIT_PAGE_CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE S+", layout="wide")

# =====================================================
# S+03 — MEMORY_PATH_MANAGER
# =====================================================

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
MEM = os.path.join(BASE_DIR, "oracle_memory")
os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": os.path.join(MEM, "fragments.csv"),
    "relations": os.path.join(MEM, "relations.json"),
    "cortex": os.path.join(MEM, "cortex.json"),
}

# =====================================================
# S+04 — SAFE_IO_LAYER
# =====================================================

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

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
        pd.DataFrame(columns=["fragment", "count"]).to_csv(
            FILES["fragments"], index=False
        )

    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})

    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS": 12,
            "age": 0,
            "new_today": 0,
            "last_day": str(datetime.date.today()),
            "timeline": [],
        })

init_memory()

# =====================================================
# B2.5 — COGNITIVE RUNTIME GUARD
# =====================================================

def runtime_id():
    if "runtime_id" not in st.session_state:
        st.session_state.runtime_id = str(uuid.uuid4())

def cognitive_clock():
    now = time.time()
    if "cognitive_time" not in st.session_state:
        st.session_state.cognitive_time = now
        st.session_state.delta_t = 0.0
    else:
        st.session_state.delta_t = now - st.session_state.cognitive_time
        st.session_state.cognitive_time = now

def runtime_guard():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.boot_time = time.time()
        st.session_state.rerun_count = 0
    else:
        st.session_state.rerun_count += 1

def cognitive_bootstrap():
    runtime_guard()
    runtime_id()
    cognitive_clock()

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
# S+17 — DELTA_K REGULATOR
# =====================================================

def delta_k_runtime():

    cortex = st.session_state.shadow_cortex

    new = cortex.get("new_today",0)
    age = max(cortex.get("age",1),1)

    ratio = new / age

    cortex["delta_ratio"] = ratio

    if ratio > 0.30:
        st.warning("⚠️ Surcharge cognitive détectée (∆k/k élevé)")

delta_k_runtime()

# =====================================================
# S+08 — TEXT_NORMALIZER
# =====================================================

def char_tokens(text):
    return [c for c in text.lower() if c.strip()]

def clean(t):
    return re.sub(r"[^\wàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

# =====================================================
# DOCX READER
# =====================================================

def read_docx(file):
    if Document is None:
        return ""
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# =====================================================
# S+09 — LEARNING_ENGINE_CORE
# =====================================================

def learn(text):

    chars = char_tokens(text)
    words = tokenize(text)
    if not words:
        return 0

    df = st.session_state.shadow_frag

    if df.empty:
        current_memory = {}
    else:
        current_memory = dict(zip(df["fragment"], df["count"]))

    counts = Counter(words)

    for w, c in counts.items():
        current_memory[w] = current_memory.get(w, 0) + c

    new_df = pd.DataFrame(
        list(current_memory.items()),
        columns=["fragment", "count"],
    )

    st.session_state.shadow_frag = new_df
    save_frag(new_df)

    assoc = st.session_state.shadow_rel

    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + 2

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

    return len(words)

# =====================================================
# S+10 — SEMANTIC METRICS
# =====================================================

def association_density():
    assoc = st.session_state.shadow_rel
    links = sum(len(v) for v in assoc.values())
    vocab = len(assoc)
    return round(links / max(vocab, 1), 2)

# =====================================================
# S+11 — PRETHINK
# =====================================================

def prethink(seed):

    assoc = st.session_state.shadow_rel

    if seed in assoc and assoc[seed]:
        return max(assoc[seed], key=assoc[seed].get)

    return seed

# =====================================================
# S+12 — THINK ENGINE
# =====================================================

def think(seed, steps=30, temp=1.0):

    assoc = st.session_state.shadow_rel

    if seed not in assoc:
        return f"Le concept '{seed}' est isolé dans ma structure."

    sent = [seed]
    cur = seed

    for _ in range(steps):

        nxt = assoc.get(cur)
        if not nxt:
            break

        w = list(nxt.keys())
        p = np.array(list(nxt.values()), dtype=float)

        p = p ** (1 / temp)
        p = p / p.sum()

        cur = np.random.choice(w, p=p)
        sent.append(cur)

    return " ".join(sent).capitalize() + "."

# =====================================================
# S+15 — USER INTERFACE
# =====================================================

st.title("🧠 ORACLE S+")

ctx = st.session_state.shadow_cortex

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitalité", round(ctx["VS"], 2))
c2.metric("Age", round(ctx["age"], 6))
c3.metric("Densité", association_density())
c4.metric("Concepts", len(st.session_state.shadow_frag))

# CHAT

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Entrez un concept..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    tokens = tokenize(prompt)

    if tokens:
        seed = prethink(tokens[0])
        response = think(seed)
    else:
        response = "Flux insuffisant."

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()

# =====================================================
# S+16.5 — MEMORY EXPORT PANEL
# =====================================================

st.divider()
st.subheader("💾 Exporter la mémoire Oracle")

col1, col2, col3 = st.columns(3)

# --- fragments.csv ---
with col1:
    if os.path.exists(FILES["fragments"]):
        with open(FILES["fragments"], "rb") as f:
            st.download_button(
                label="⬇️ fragments.csv",
                data=f,
                file_name="oracle_fragments.csv",
                mime="text/csv",
                use_container_width=True,
            )

# --- relations.json ---
with col2:
    if os.path.exists(FILES["relations"]):
        with open(FILES["relations"], "rb") as f:
            st.download_button(
                label="⬇️ relations.json",
                data=f,
                file_name="oracle_relations.json",
                mime="application/json",
                use_container_width=True,
            )

# --- cortex.json ---
with col3:
    if os.path.exists(FILES["cortex"]):
        with open(FILES["cortex"], "rb") as f:
            st.download_button(
                label="⬇️ cortex.json",
                data=f,
                file_name="oracle_cortex.json",
                mime="application/json",
                use_container_width=True,
            )

# =====================================================
# S+16.6 — MEMORY IMPORT PANEL
# =====================================================

st.subheader("📥 Importer une mémoire Oracle")

uploaded = st.file_uploader(
    "Importer fragments / relations / cortex ou un ZIP complet",
    type=["csv", "json", "zip"],
)

import zipfile
import io

def safe_reload():
    # recharge le shadow state après import
    st.session_state.shadow_loaded = False
    sync_shadow()

if uploaded is not None:

    try:

        # ==============================
        # CAS 1 — ZIP COMPLET
        # ==============================
        if uploaded.name.endswith(".zip"):

            z = zipfile.ZipFile(io.BytesIO(uploaded.read()))

            for name in z.namelist():

                if "fragments" in name:
                    with open(FILES["fragments"], "wb") as f:
                        f.write(z.read(name))

                elif "relations" in name:
                    with open(FILES["relations"], "wb") as f:
                        f.write(z.read(name))

                elif "cortex" in name:
                    with open(FILES["cortex"], "wb") as f:
                        f.write(z.read(name))

            safe_reload()
            st.success("✅ Mémoire complète importée.")

        # ==============================
        # CAS 2 — CSV (fragments)
        # ==============================
        elif uploaded.name.endswith(".csv"):

            df = pd.read_csv(uploaded)
            save_frag(df)

            safe_reload()
            st.success("✅ fragments.csv importé.")

        # ==============================
        # CAS 3 — JSON
        # ==============================
        elif uploaded.name.endswith(".json"):

            data = json.load(uploaded)

            # détection automatique
            if "timeline" in data:
                save_json(FILES["cortex"], data)
                st.success("✅ cortex.json importé.")
            else:
                save_json(FILES["relations"], data)
                st.success("✅ relations.json importé.")

            safe_reload()

    except Exception as e:
        st.error(f"❌ Import impossible : {e}")

# =====================================================
# S+16 — UI RENDER
# =====================================================

st.caption(
    f"Temps cognitif : {round(st.session_state.delta_t,2)} s"
)

spectral_ui(
    st.session_state.shadow_cortex,
    st.session_state.shadow_frag["fragment"].tolist()
)