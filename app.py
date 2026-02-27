# =====================================================
# 🧠 TTU ORACLE V11 — SHADOW STATE FINAL STABLE
# TST Ghost Memory + Auto Diagnostic AI
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

MEM="oracle_memory"
os.makedirs(MEM,exist_ok=True)

FILES={
    "fragments":f"{MEM}/fragments.csv",
    "concepts":f"{MEM}/concepts.csv",
    "relations":f"{MEM}/relations.json",
    "intentions":f"{MEM}/intentions.csv",
    "cortex":f"{MEM}/cortex.json"
}

# --------------------------------------------------
# INIT
# --------------------------------------------------

def init():

    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)

    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept","weight"]).to_csv(FILES["concepts"],index=False)

    if not os.path.exists(FILES["intentions"]):
        pd.DataFrame(columns=["intent","count"]).to_csv(FILES["intentions"],index=False)

    if not os.path.exists(FILES["relations"]):
        json.dump({},open(FILES["relations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "VS":12,
            "age":0,
            "new_today":0,
            "last_day":str(datetime.date.today())
        },open(FILES["cortex"],"w"))

init()

# --------------------------------------------------
# LOAD SAVE SAFE
# --------------------------------------------------

def load_json(p):
    with open(p,"r") as f:
        return json.load(f)

def save_json(p,d):
    with open(p,"w") as f:
        json.dump(d,f)

def load_frag(): return pd.read_csv(FILES["fragments"])
def save_frag(df): df.reset_index(drop=True).to_csv(FILES["fragments"],index=False)

def load_concepts(): return pd.read_csv(FILES["concepts"])
def save_concepts(df): df.reset_index(drop=True).to_csv(FILES["concepts"],index=False)

# --------------------------------------------------
# 👻 SHADOW LOAD
# --------------------------------------------------

def sync_shadow():

    if "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_concepts = load_concepts().copy()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])

        st.session_state.shadow_loaded=True

sync_shadow()

# --------------------------------------------------
# FAST METRICS
# --------------------------------------------------

def association_density_fast():
    assoc=st.session_state.shadow_rel
    links=sum(len(v) for v in assoc.values())
    vocab=len(assoc)
    return round(links/max(vocab,1),2)

def semantic_coherence_fast():
    concepts=len(st.session_state.shadow_concepts)
    assoc=len(st.session_state.shadow_rel)
    return round(min(100,(assoc/max(concepts,1))*10),2)

# --------------------------------------------------
# LANGUAGE CORE
# --------------------------------------------------

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

# --------------------------------------------------
# FILE READER SAFE
# --------------------------------------------------

def read_file(file):

    name=file.name.lower()

    try:

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

    except:
        return ""

    return ""

# --------------------------------------------------
# 🧠 LEARN ENGINE (SAFE)
# --------------------------------------------------

def learn(text):

    words=tokenize(text)
    if not words:
        return 0

    df=st.session_state.shadow_frag.copy()
    counts=Counter(words)

    for w,c in counts.items():

        mask=df["fragment"]==w

        if mask.any():
            df.loc[mask,"count"]+=c
        else:
            df=pd.concat(
                [df,pd.DataFrame([[w,c]],columns=df.columns)],
                ignore_index=True
            )

    assoc=st.session_state.shadow_rel

    for i in range(len(words)-1):
        a,b=words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+2

    cortex=st.session_state.shadow_cortex
    today=str(datetime.date.today())

    if cortex["last_day"]!=today:
        cortex["new_today"]=0
        cortex["last_day"]=today

    cortex["age"]+=len(words)
    cortex["new_today"]+=len(counts)
    cortex["VS"]=10+float(np.log1p(cortex["age"]))

    save_frag(df)
    save_json(FILES["relations"],assoc)
    save_json(FILES["cortex"],cortex)

    st.session_state.shadow_frag=df

    return len(words)

# --------------------------------------------------
# THINK ENGINE STABLE
# --------------------------------------------------

def think(seed,steps=30):

    assoc=st.session_state.shadow_rel

    if seed not in assoc:
        return "Je dois encore apprendre sur ce concept."

    sent=[seed]
    cur=seed

    for _ in range(steps):

        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)

        s=p.sum()
        if s==0: break

        p=p/s
        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."

# --------------------------------------------------
# AUTO DIAG
# --------------------------------------------------

def diagnose():

    cortex=st.session_state.shadow_cortex
    density=association_density_fast()

    if cortex["new_today"]<20:
        return "🧠 J'ai besoin de nouvelles connaissances."

    if density<1.5:
        return "🧠 Donne-moi des textes plus longs."

    if density>4:
        return "🧠 Mon raisonnement commence à émerger."

    return "🧠 Apprentissage actif."

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🧠 ORACLE V11 — SHADOW STATE")

ctx=st.session_state.shadow_cortex

c1,c2,c3,c4=st.columns(4)
c1.metric("Vitalité Spectrale",round(ctx["VS"],2))
c2.metric("Âge Cognitif",ctx["age"])
c3.metric("Densité Associative",association_density_fast())
c4.metric("Cohérence %",semantic_coherence_fast())

st.info(diagnose())

# --------------------------------------------------
# LEARNING
# --------------------------------------------------

st.subheader("📥 Nourrir l'IA")

file=st.file_uploader(
"Nourriture cognitive",
type=["txt","csv","pdf","docx","xlsx"]
)

if file:
    text=read_file(file)
    n=learn(text)
    st.success(f"{n} unités cognitives assimilées")

# --------------------------------------------------
# CHAT
# --------------------------------------------------

st.subheader("💬 Dialogue cognitif")

prompt=st.text_input("Intention")

if st.button("Penser"):

    tokens=tokenize(prompt)

    if not tokens:
        st.warning("Entre une phrase valide.")
    else:
        st.write("### Réponse")
        st.write(think(tokens[0]))
