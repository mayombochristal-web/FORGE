# =====================================================
# 🧠 ORACLE S+ v3 — ARCHITECTURE COGNITIVE STABLE
# Pipeline officiel : S+01 → S+17
# =====================================================

# =====================================================
# S+01 — IMPORT_SYSTEM_CORE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, datetime, time, uuid, glob, zipfile, io
from collections import Counter

try:
    from spectral_module import spectral_ui
except:
    def spectral_ui(*args, **kwargs):
        pass

try:
    from docx import Document
except:
    Document=None

try:
    from pypdf import PdfReader
except:
    PdfReader=None


# =====================================================
# S+02 — STREAMLIT_PAGE_CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE S+", layout="wide")


# =====================================================
# S+03 — MEMORY_PATH_MANAGER
# =====================================================

BASE_DIR="."
MEM=os.path.join(BASE_DIR,"oracle_memory")
os.makedirs(MEM,exist_ok=True)

FILES={
 "fragments":os.path.join(MEM,"fragments.csv"),
 "relations":os.path.join(MEM,"relations.json"),
 "cortex":os.path.join(MEM,"cortex.json"),
}


# =====================================================
# S+04 — SAFE_IO_LAYER (⚠️ NE PAS CASSER)
# =====================================================

def load_json(p):
    try:
        with open(p,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_json(p,d):
    with open(p,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False,indent=2)

def load_frag():
    if os.path.exists(FILES["fragments"]):
        return pd.read_csv(FILES["fragments"])
    return pd.DataFrame(columns=["fragment","count"])

def save_frag(df):
    df.to_csv(FILES["fragments"],index=False)


# =====================================================
# S+05 — MEMORY_INITIALIZER
# =====================================================

def init_memory():

    if not os.path.exists(FILES["fragments"]):
        save_frag(pd.DataFrame(columns=["fragment","count"]))

    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"],{})

    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"],{
            "VS":12,
            "age":0,
            "new_today":0,
            "last_day":str(datetime.date.today()),
            "timeline":[]
        })

init_memory()


# =====================================================
# S+06 — SHADOW_STATE_LOADER
# =====================================================

def sync_shadow(force=False):

    if force or "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag=load_frag().copy()
        st.session_state.shadow_rel=load_json(FILES["relations"])
        st.session_state.shadow_cortex=load_json(FILES["cortex"])

        st.session_state.shadow_loaded=True

def safe_reload():
    sync_shadow(True)

sync_shadow()


# =====================================================
# S+07 — COGNITIVE_TIME_TRACKER
# =====================================================

if "t0" not in st.session_state:
    st.session_state.t0=time.time()

st.session_state.cognitive_time=time.time()-st.session_state.t0


# =====================================================
# S+08 — TEXT_NORMALIZER
# =====================================================

def clean(t):
    return re.sub(r"[^\wàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]


# =====================================================
# S+08.5 — FILE READERS
# =====================================================

def read_docx(file):
    if Document is None: return ""
    doc=Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file):
    if PdfReader is None: return ""
    try:
        reader=PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


# =====================================================
# S+09 — LEARNING_ENGINE_CORE
# =====================================================

def learn(text):

    words=tokenize(text)
    if not words: return 0

    df=st.session_state.shadow_frag.copy()
    memory=dict(zip(df["fragment"],df["count"]))

    for w,c in Counter(words).items():
        memory[w]=memory.get(w,0)+c

    new_df=pd.DataFrame(memory.items(),
                        columns=["fragment","count"])

    save_frag(new_df)
    st.session_state.shadow_frag=new_df

    assoc=st.session_state.shadow_rel

    for i in range(len(words)-1):
        a,b=words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+2

    save_json(FILES["relations"],assoc)

    cortex=st.session_state.shadow_cortex
    cortex["age"]+=len(words)
    cortex["VS"]=10+float(np.log1p(cortex["age"]))
    cortex["timeline"].extend(words[-200:])

    save_json(FILES["cortex"],cortex)

    safe_reload()
    return len(words)


# =====================================================
# S+10 — SEMANTIC_SEARCH_ENGINE
# =====================================================

def association_density():
    assoc=st.session_state.shadow_rel
    links=sum(len(v) for v in assoc.values())
    return round(links/max(len(assoc),1),2)


# =====================================================
# S+11 — PRETHINK_ENGINE
# =====================================================

def prethink(seed):
    assoc=st.session_state.shadow_rel
    if seed in assoc and assoc[seed]:
        return max(assoc[seed],key=assoc[seed].get)
    return seed


# =====================================================
# S+12 — THINK_GENERATION_ENGINE
# =====================================================

def think(seed,steps=30):

    assoc=st.session_state.shadow_rel
    if seed not in assoc:
        return "Je dois encore apprendre."

    sent=[seed]
    cur=seed

    for _ in range(steps):

        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."


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
    d=association_density()
    if d<1.5: return "🧠 Apprends encore."
    if d>4: return "🧠 Émergence cognitive."
    return "🧠 Apprentissage actif."


# =====================================================
# S+17 — CACHE COGNITIF (🔥 NOUVEAU)
# =====================================================

@st.cache_data(show_spinner=False,ttl=600)
def cached_think(seed,steps,rel_snapshot):

    # rel_snapshot = hash cognitif
    np.random.seed(abs(hash(seed))%2**32)

    assoc=rel_snapshot

    if seed not in assoc:
        return "Je dois encore apprendre."

    sent=[seed]
    cur=seed

    for _ in range(steps):

        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."


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

uploaded=st.file_uploader(
    "Nourrir l'IA",
    type=["txt","csv","docx","pdf"]
)

if uploaded:

    if uploaded.name.endswith(".docx"):
        text=read_docx(uploaded)
    elif uploaded.name.endswith(".pdf"):
        text=read_pdf(uploaded)
    else:
        text=uploaded.read().decode("utf-8","ignore")

    n=learn(text)
    st.success(f"{n} unités assimilées")


prompt=st.text_input("Intention cognitive")

if st.button("Penser ⚡"):

    tokens=tokenize(prompt)

    if tokens:
        seed=prethink(tokens[0])

        # snapshot = clé du cache
        rel_snapshot=st.session_state.shadow_rel.copy()

        response=cached_think(seed,30,rel_snapshot)

        st.success("⚡ Réponse instantanée")
        st.write(response)
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