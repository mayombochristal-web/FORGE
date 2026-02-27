# =====================================================
# 🧠 TTU ORACLE V11 — SHADOW STATE AUTO-ÉVOLUTION
# État Fantôme TST + IA auto-diagnostique
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

st.set_page_config(page_title="ORACLE V11 SHADOW", layout="wide")

# --------------------------------------------------
# MEMORY STRUCTURE
# --------------------------------------------------

MEM = "oracle_memory"
os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "concepts": f"{MEM}/concepts.csv",
    "relations": f"{MEM}/relations.json",
    "intentions": f"{MEM}/intentions.csv",
    "cortex": f"{MEM}/cortex.json"
}

# --------------------------------------------------
# INIT MEMORY
# --------------------------------------------------

def init():

    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)

    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept","weight"]).to_csv(FILES["concepts"],index=False)

    if not os.path.exists(FILES["intentions"]):
        pd.DataFrame(columns=["intent","count"]).to_csv(FILES["intentions"],index=False)

    if not os.path.exists(FILES["relations"]):
        json.dump({}, open(FILES["relations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "VS":12,
            "age":0,
            "new_today":0,
            "last_day":str(datetime.date.today())
        }, open(FILES["cortex"],"w"))

init()

# --------------------------------------------------
# LOAD / SAVE
# --------------------------------------------------

def load_json(p): return json.load(open(p))
def save_json(p,d): json.dump(d,open(p,"w"))

def load_frag(): return pd.read_csv(FILES["fragments"])
def save_frag(df): df.to_csv(FILES["fragments"],index=False)

def load_concepts(): return pd.read_csv(FILES["concepts"])
def save_concepts(df): df.to_csv(FILES["concepts"],index=False)

# --------------------------------------------------
# 👻 SHADOW STATE (TST ANCRAGE)
# --------------------------------------------------

def sync_shadow():

    if "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag = load_frag()
        st.session_state.shadow_concepts = load_concepts()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])

        st.session_state.shadow_loaded = True

sync_shadow()

# --------------------------------------------------
# FAST METRICS (RAM ONLY)
# --------------------------------------------------

def association_density_fast():

    assoc = st.session_state.shadow_rel
    total_links = sum(len(v) for v in assoc.values())
    vocab = len(assoc)

    return round(total_links / max(vocab,1),2)

def semantic_coherence_fast():

    concepts = len(st.session_state.shadow_concepts)
    assoc = len(st.session_state.shadow_rel)

    return round(min(100,(assoc/max(concepts,1))*10),2)

# --------------------------------------------------
# LANGUAGE CORE
# --------------------------------------------------

VOWELS="aeiouyàâéèêëîïôùûüœ"

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

# --------------------------------------------------
# FILE READER UNIVERSAL
# --------------------------------------------------

def read_file(file):

    name=file.name.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8","ignore")

    if name.endswith(".csv"):
        return pd.read_csv(file).to_string()

    if name.endswith(".xlsx"):
        return pd.read_excel(file).to_string()

    if name.endswith(".docx"):
        doc=zipfile.ZipFile(io.BytesIO(file.read()))
        xml=doc.read("word/document.xml")
        tree=ET.fromstring(xml)
        return " ".join(t.text for t in tree.iter() if t.text)

    if name.endswith(".pdf"):
        return file.read().decode("latin-1","ignore")

    return ""

# --------------------------------------------------
# 🧠 LEARN (WRITE-BACK SHADOW)
# --------------------------------------------------

def learn(text):

    words = tokenize(text)
    fragments = Counter(words)

    df = st.session_state.shadow_frag

    for w,c in fragments.items():
        if w in df.fragment.values:
            df.loc[df.fragment==w,"count"]+=c
        else:
            df.loc[len(df)]=[w,c]

    assoc = st.session_state.shadow_rel

    for i in range(len(words)-1):
        a,b = words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+2

    cortex = st.session_state.shadow_cortex

    today=str(datetime.date.today())
    if cortex["last_day"]!=today:
        cortex["new_today"]=0
        cortex["last_day"]=today

    cortex["age"]+=len(words)
    cortex["new_today"]+=len(fragments)
    cortex["VS"]=10+np.log1p(cortex["age"])

    # WRITE BACK DISK
    save_frag(df)
    save_json(FILES["relations"],assoc)
    save_json(FILES["cortex"],cortex)

    return len(words)

# --------------------------------------------------
# THINK ENGINE (RAM ONLY)
# --------------------------------------------------

def think(seed,steps=30):

    assoc = st.session_state.shadow_rel

    if seed not in assoc:
        return "Je dois encore apprendre sur ce concept."

    sent=[seed]
    cur=seed

    for _ in range(steps):

        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()))
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."

# --------------------------------------------------
# 🧠 AUTO DIAGNOSTIC (IA QUI DEMANDE À APPRENDRE)
# --------------------------------------------------

def diagnose():

    cortex = st.session_state.shadow_cortex
    density = association_density_fast()

    if cortex["new_today"] < 20:
        return "🧠 J'ai besoin de nouvelles connaissances (PDF, dialogues)."

    if density < 1.5:
        return "🧠 J'ai besoin de plus de contexte (textes longs)."

    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."

    return "🧠 Je suis en phase d'apprentissage actif."

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🧠 ORACLE V11 — SHADOW STATE")

ctx = st.session_state.shadow_cortex

c1,c2,c3,c4 = st.columns(4)

c1.metric("Vitalité Spectrale",round(ctx["VS"],2))
c2.metric("Âge Cognitif",ctx["age"])
c3.metric("Densité Associative",association_density_fast())
c4.metric("Cohérence %",semantic_coherence_fast())

st.info(diagnose())

# --------------------------------------------------
# LEARN
# --------------------------------------------------

st.subheader("📥 Nourrir l'IA")

file = st.file_uploader(
"Nourriture cognitive",
type=["txt","csv","pdf","docx","xlsx"]
)

if file:
    text = read_file(file)
    n = learn(text)
    st.success(f"{n} unités cognitives assimilées")

# --------------------------------------------------
# CHAT
# --------------------------------------------------

st.subheader("💬 Dialogue cognitif")

prompt = st.text_input("Intention")

if st.button("Penser"):
    if prompt:
        st.write("### Réponse")
        st.write(think(tokenize(prompt)[0]))
