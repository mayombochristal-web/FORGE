# =====================================================
# 🧠 TTU ORACLE V10 — AUTO-ÉVOLUTION STABLE
# Mémoire cognitive persistante + diagnostic IA
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

st.set_page_config(page_title="ORACLE V10 AUTO", layout="wide")

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
"cortex":f"{MEM}/cortex.json",
"index":f"{MEM}/knowledge_index.json"
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
        json.dump({},open(FILES["relations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({"VS":12,"age":0,"last_day":str(datetime.date.today()),"new_today":0},open(FILES["cortex"],"w"))

    if not os.path.exists(FILES["index"]):
        json.dump({"files":0},open(FILES["index"],"w"))

init()

# --------------------------------------------------
# LOAD SAVE
# --------------------------------------------------

def load_json(p): return json.load(open(p))
def save_json(p,d): json.dump(d,open(p,"w"))

def load_frag(): return pd.read_csv(FILES["fragments"])
def save_frag(df): df.to_csv(FILES["fragments"],index=False)

def load_concepts(): return pd.read_csv(FILES["concepts"])
def save_concepts(df): df.to_csv(FILES["concepts"],index=False)

# --------------------------------------------------
# FILE READER (PDF WORD FIX)
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
        # lecture brute stable cloud
        return file.read().decode("latin-1","ignore")

    return ""

# --------------------------------------------------
# LANGUAGE CORE
# --------------------------------------------------

VOWELS="aeiouyàâéèêëîïôùûüœ"

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

def syllabify(word):

    syll=[]
    cur=""

    for c in word:
        cur+=c
        if c in VOWELS:
            syll.append(cur)
            cur=""

    if cur:
        if syll:
            syll[-1]+=cur
        else:
            syll=[cur]

    return syll

# --------------------------------------------------
# LEARNING ENGINE
# --------------------------------------------------

def learn(text):

    words=tokenize(text)
    fragments=Counter(words)

    df=load_frag()

    for w,c in fragments.items():
        if w in df.fragment.values:
            df.loc[df.fragment==w,"count"]+=c
        else:
            df.loc[len(df)]=[w,c]

    save_frag(df)

    # ---- concepts
    concepts=load_concepts()
    for w in fragments:
        if w in concepts.concept.values:
            concepts.loc[concepts.concept==w,"weight"]+=1
        else:
            concepts.loc[len(concepts)]=[w,1]

    save_concepts(concepts)

    # ---- associations
    assoc=load_json(FILES["relations"])

    for i in range(len(words)-1):
        a,b=words[i],words[i+1]
        assoc.setdefault(a,{})
        assoc[a][b]=assoc[a].get(b,0)+2

    save_json(FILES["relations"],assoc)

    # ---- cortex update
    cortex=load_json(FILES["cortex"])

    today=str(datetime.date.today())
    if cortex["last_day"]!=today:
        cortex["new_today"]=0
        cortex["last_day"]=today

    cortex["age"]+=len(words)
    cortex["new_today"]+=len(fragments)
    cortex["VS"]=10+np.log1p(cortex["age"])

    save_json(FILES["cortex"],cortex)

    return len(words)

# --------------------------------------------------
# THINKING
# --------------------------------------------------

def think(seed,steps=30):

    assoc=load_json(FILES["relations"])

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
# METRICS
# --------------------------------------------------

def association_density():
    assoc=load_json(FILES["relations"])
    total_links=sum(len(v) for v in assoc.values())
    vocab=len(assoc)
    return round(total_links/max(vocab,1),2)

def semantic_coherence():
    concepts=len(load_concepts())
    assoc=len(load_json(FILES["relations"]))
    return round(min(100,(assoc/max(concepts,1))*10),2)

def diagnose():

    cortex=load_json(FILES["cortex"])
    density=association_density()

    if cortex["new_today"]<20:
        return "🧠 Besoin : NOUVELLES DONNÉES"

    if density<1.5:
        return "🧠 Besoin : PLUS DE CONTEXTE"

    if density>4:
        return "🧠 État : RAISONNEMENT ÉMERGENT"

    return "🧠 État : APPRENTISSAGE ACTIF"

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🧠 ORACLE V10 — AUTO-ÉVOLUTION")

cortex=load_json(FILES["cortex"])

c1,c2,c3,c4=st.columns(4)

c1.metric("Vitalité Spectrale",round(cortex["VS"],2))
c2.metric("Âge cognitif",cortex["age"])
c3.metric("Densité associative",association_density())
c4.metric("Cohérence %",semantic_coherence())

st.info(diagnose())

# --------------------------------------------------
# LEARN
# --------------------------------------------------

st.subheader("📥 Nourrir l'IA")

file=st.file_uploader(
"Importer connaissance",
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
    if prompt:
        st.write("### Réponse")
        st.write(think(tokenize(prompt)[0]))
