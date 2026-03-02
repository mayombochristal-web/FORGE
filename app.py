# =====================================================
# 🧠 ORACLE V4 FULL — V3.1 + EXTENSION COGNITIVE
# Toutes entrées V2/V3.1 conservées
# =====================================================

import streamlit as st
import random, json, os, math, time, io
import PyPDF2, docx
import pandas as pd
import speech_recognition as sr
from collections import deque, Counter

# =====================================================
# CONFIGURATION
# =====================================================

MEM_DIR="oracle_memory"
LEXICON_PATH=os.path.join(MEM_DIR,"lexicon.json")
SELF_PATH=os.path.join(MEM_DIR,"self_model.json")

os.makedirs(MEM_DIR,exist_ok=True)

def init_file(p,d):
    if not os.path.exists(p):
        json.dump(d,open(p,"w",encoding="utf-8"))

init_file(LEXICON_PATH,{})
init_file(SELF_PATH,{
    "cognitive_age":0,
    "stability":0.5,
    "interactions":0
})

# =====================================================
# SESSION
# =====================================================

if "phi" not in st.session_state:
    st.session_state.phi={"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory=deque(maxlen=40)

if "green_state" not in st.session_state:
    st.session_state.green_state=0.0

if "last_sleep" not in st.session_state:
    st.session_state.last_sleep=time.time()

# =====================================================
# GREEN NOISE
# =====================================================

def green_noise(prev):
    return 0.92*prev+(1-0.92)*random.uniform(-1,1)

def consolidation_gate():
    st.session_state.green_state=green_noise(
        st.session_state.green_state)
    return abs(st.session_state.green_state)<0.25

# =====================================================
# Φ ENGINE
# =====================================================

def evolve_phi(phi,exc):

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    s=sum(phi.values())
    for k in phi:
        phi[k]/=s
    return phi

# =====================================================
# MEMORY IO
# =====================================================

def load_json(p):
    try:
        return json.load(open(p,"r",encoding="utf-8"))
    except:
        return {}

def save_json(p,d):
    json.dump(d,open(p,"w",encoding="utf-8"),
              indent=2,ensure_ascii=False)

# =====================================================
# HIPPOCAMPUS (concept hierarchy)
# =====================================================

def concept_tag(w):
    return w[:4] if len(w)>7 else w

# =====================================================
# LEARNING
# =====================================================

def learn(text,phi,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_json(LEXICON_PATH)
    energy=math.sqrt(sum(v*v for v in phi.values()))*importance

    for a,b in zip(words,words[1:]):
        a,b=concept_tag(a),concept_tag(b)
        L.setdefault(a,{})
        L[a][b]=L[a].get(b,0)+energy

    if consolidation_gate():
        save_json(LEXICON_PATH,L)

# =====================================================
# SOMMEIL
# =====================================================

def sleep_cycle():

    L=load_json(LEXICON_PATH)
    new={}

    for w,con in L.items():
        f={t:v*0.997 for t,v in con.items() if v>1.2}
        if f:
            new[w]=f

    save_json(LEXICON_PATH,new)

    self_model=load_json(SELF_PATH)
    self_model["cognitive_age"]+=1
    self_model["stability"]=min(1,self_model["stability"]+0.01)
    save_json(SELF_PATH,self_model)

    st.session_state.last_sleep=time.time()
    return "🌙 Consolidation terminée"

# =====================================================
# META-COGNITION
# =====================================================

def evaluate_response(words):
    return len(set(words))/max(len(words),1)

# =====================================================
# THALAMUS
# =====================================================

def contextual_seed(L):

    context=" ".join(st.session_state.dialog_memory).split()
    c=[concept_tag(w) for w in context if concept_tag(w) in L]

    if c:
        return Counter(c).most_common(1)[0][0]

    return random.choice(list(L.keys()))

# =====================================================
# CORTEX
# =====================================================

def logical_layer(seq,L):
    return [w for w in seq if w in L] or seq

def associative_layer(word,L,phi):

    if word not in L:
        return word

    opts=L[word]

    if random.random()<phi["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

def predictive_layer(words,phi):

    if phi["phi_m"]>0.6:
        words.append("continuité")

    if phi["phi_c"]>0.65:
        words.append("évolution")

    return words

# =====================================================
# GENERATION
# =====================================================

def oracle_reply(phi):

    L=load_json(LEXICON_PATH)
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(L)
    words=[seed]
    used=set(words)

    for _ in range(int(8+phi["phi_m"]*35)):

        filtered=logical_layer(words,L)
        current=filtered[-1]

        nxt=associative_layer(current,L,phi)

        if nxt in used and random.random()<phi["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    words=predictive_layer(words,phi)

    if evaluate_response(words)<0.35:
        return "Je reformule ma pensée."

    return " ".join(words).capitalize()+"."

# =====================================================
# EXTRACTION MULTIMODALE (V2 RESTAURÉ)
# =====================================================

def extract_content(mode):

    raw=""

    if mode=="Texte":
        raw=st.text_area("Texte")

    elif mode=="Document":
        file=st.file_uploader(
            "PDF / DOCX / TXT",
            type=["pdf","docx","txt"]
        )

        if file:

            if file.name.endswith(".pdf"):
                reader=PyPDF2.PdfReader(file)
                raw=" ".join(
                    p.extract_text() for p in reader.pages
                    if p.extract_text()
                )

            elif file.name.endswith(".docx"):
                buffer=io.BytesIO(file.read())
                doc=docx.Document(buffer)
                raw="\n".join(p.text for p in doc.paragraphs)

            else:
                raw=file.read().decode("utf-8")

    elif mode=="Excel":

        file=st.file_uploader("Excel",type=["xlsx","xls"])
        if file:
            df=pd.read_excel(file)
            raw=df.to_string()

    elif mode=="Audio":

        audio=st.file_uploader("Audio WAV",type="wav")
        if audio:
            r=sr.Recognizer()
            with sr.AudioFile(audio) as source:
                data=r.record(source)
                try:
                    raw=r.recognize_google(data,language="fr-FR")
                    st.success("Transcription réussie")
                except:
                    st.error("Transcription impossible")

    return raw

# =====================================================
# AUTO SOMMEIL
# =====================================================

def auto_sleep():
    if time.time()-st.session_state.last_sleep>180:
        if consolidation_gate():
            sleep_cycle()

auto_sleep()

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V4",page_icon="🧠")

st.title("🧠 ORACLE V4 — Agent Cognitif Stable")

mode=st.radio(
    "Source du savoir",
    ["Texte","Document","Excel","Audio"]
)

content=extract_content(mode)

if st.button("🌱 Nourrir") and content:
    exc=min(1,len(content)/400)
    st.session_state.phi=evolve_phi(
        st.session_state.phi,exc)
    learn(content,st.session_state.phi,1.2)

msg=st.text_input("Parlez")

if st.button("➡️") and msg:

    st.session_state.dialog_memory.append(msg)

    learn(msg,st.session_state.phi,1.1)

    reply=oracle_reply(st.session_state.phi)

    st.session_state.dialog_memory.append(reply)

for m in st.session_state.dialog_memory:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    size=os.path.getsize(LEXICON_PATH)/1024
    st.success(f"Mémoire : {size:.2f} KB")

    self_model=load_json(SELF_PATH)
    st.write("Âge cognitif :",self_model["cognitive_age"])
    st.progress(self_model["stability"])

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}:{v:.2f}")

    if st.button("🌙 Sommeil forcé"):
        st.warning(sleep_cycle())
        st.rerun()