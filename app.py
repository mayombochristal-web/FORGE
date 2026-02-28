# =====================================================
# 🧠 ORACLE V13 — TTU EMERGENT LLM LOCAL
# (V11 + V12 fusion NON DESTRUCTIVE)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE V13 TTU", layout="wide")

MEM="oracle_memory"
os.makedirs(MEM,exist_ok=True)

FILES={
 "fragments":f"{MEM}/fragments.csv",
 "concepts":f"{MEM}/concepts.csv",
 "relations":f"{MEM}/relations.json",
 "cortex":f"{MEM}/cortex.json"
}

# =====================================================
# SAFE LOADERS
# =====================================================

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def save_json(path,data):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def lazy_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    chunks=[]
    for c in pd.read_csv(path,chunksize=5000):
        chunks.append(c)
    return pd.concat(chunks,ignore_index=True)

def save_csv(df,path):
    df.reset_index(drop=True).to_csv(path,index=False)

# =====================================================
# INIT MEMORY
# =====================================================

def init():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)

    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"],{})

    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"],{
            "age":0,
            "VS":12,
            "timeline":[],
            "temperature":0.4,
            "IS":1.0
        })

init()

# =====================================================
# LOADERS SPECIALISES
# =====================================================

def load_frag(FILES):
    return lazy_csv(FILES["fragments"])

def save_frag(df,FILES):
    save_csv(df,FILES["fragments"])

# =====================================================
# SHADOW SESSION
# =====================================================

def sync_shadow():
    if "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag = load_frag(FILES)
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])

        st.session_state.shadow_loaded=True

sync_shadow()

# =====================================================
# NLP CORE
# =====================================================

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

# =====================================================
# LEARNING ENGINE (fusion V11+V12)
# =====================================================

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
            df=pd.concat([df,pd.DataFrame([[w,c]],columns=df.columns)])

    save_frag(df,FILES)
    st.session_state.shadow_frag=df

    assoc=st.session_state.shadow_rel

    for i in range(len(words)-1):
        a,b=words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+1

    save_json(FILES["relations"],assoc)

    cortex=st.session_state.shadow_cortex

    for w in words:
        cortex["timeline"].append(w)

    cortex["age"]+=len(words)
    cortex["VS"]=10+np.log1p(cortex["age"])

    save_json(FILES["cortex"],cortex)

    return len(words)

# =====================================================
# 🧠 SEMANTIC VECTOR SPACE (LLM CORE)
# =====================================================

def build_semantic_vectors(window=2):

    assoc=st.session_state.shadow_rel
    vocab=list(assoc.keys())

    index={w:i for i,w in enumerate(vocab)}
    M=np.zeros((len(vocab),len(vocab)))

    for w,links in assoc.items():
        for v,val in links.items():
            if v in index:
                M[index[w],index[v]]=val

    return vocab,M

def semantic_search(query,topk=5):

    vocab,M=build_semantic_vectors()

    if query not in vocab:
        return []

    idx=vocab.index(query)
    q=M[idx]

    sims=[]
    for i,row in enumerate(M):
        sim=np.dot(q,row)/(np.linalg.norm(q)*np.linalg.norm(row)+1e-9)
        sims.append((vocab[i],sim))

    sims.sort(key=lambda x:x[1],reverse=True)

    return sims[1:topk+1]

# =====================================================
# THINK ENGINE (TEMPÉRATURE TTU)
# =====================================================

def think(seed,steps=30):

    assoc=st.session_state.shadow_rel
    T=st.session_state.shadow_cortex["temperature"]

    if seed not in assoc:
        return "Je dois encore apprendre."

    sent=[seed]
    cur=seed

    for _ in range(steps):
        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)

        p=p**(1/(T+0.01))
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."

# =====================================================
# METRICS
# =====================================================

def association_density():
    assoc=st.session_state.shadow_rel
    links=sum(len(v) for v in assoc.values())
    return round(links/max(len(assoc),1),2)

# =====================================================
# UI
# =====================================================

st.title("🧠 ORACLE V13 — LLM TTU ÉMERGENT")

ctx=st.session_state.shadow_cortex

c1,c2,c3=st.columns(3)
c1.metric("Vitalité",round(ctx["VS"],2))
c2.metric("Age Cognitif",ctx["age"])
c3.metric("Densité",association_density())

ctx["temperature"]=st.slider(
 "Température cognitive",0.05,1.5,ctx["temperature"]
)

# ================= LEARN =================

st.subheader("📥 Nourrir l'IA")

file=st.file_uploader("Texte",type=["txt","csv","docx","pdf"])

if file:
    text=file.read().decode("utf-8","ignore")
    n=learn(text)
    st.success(f"{n} unités cognitives assimilées")

# ================= CHAT =================

st.subheader("💬 Dialogue")

prompt=st.text_input("Intention")

if st.button("Penser"):
    st.write(think(prompt.lower()))

# ================= SEMANTIC SEARCH =================

st.subheader("🔎 Recherche sémantique réelle")

query=st.text_input("Concept à explorer")

if st.button("Explorer"):
    results=semantic_search(query.lower())

    if results:
        for w,s in results:
            st.write(f"{w} → similarité {round(s,3)}")
    else:
        st.info("Concept inconnu.")
