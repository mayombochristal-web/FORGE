=====================================================

🧠 ORACLE V4.5 Ω — AGENT COGNITIF COMPLET FINAL

Fusion V2 + V3.1 + V4 + V4.5

=====================================================

import streamlit as st
import random, json, os, math, time, base64
import requests
import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr
from collections import deque, Counter

=====================================================

CONFIGURATION

=====================================================

MEM_FILE = "oracle_memory.json"

if not os.path.exists(MEM_FILE):
json.dump({}, open(MEM_FILE, "w", encoding="utf-8"))

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
BRANCH = "main"

=====================================================

SESSION STATE

=====================================================

if "phi" not in st.session_state:
st.session_state.phi = {"phi_m":0.5,"phi_c":0.5,"phi_d":0.5}

if "dialog" not in st.session_state:
st.session_state.dialog = deque(maxlen=60)

if "hippocampus" not in st.session_state:
st.session_state.hippocampus = []

if "green_state" not in st.session_state:
st.session_state.green_state = 0.0

if "last_sleep" not in st.session_state:
st.session_state.last_sleep = time.time()

=====================================================

GREEN NOISE — HOMEOSTASIS

=====================================================

def green_noise(prev):
return 0.92 * prev + 0.08 * random.uniform(-1,1)

def consolidation_gate():
st.session_state.green_state = green_noise(
st.session_state.green_state
)
return abs(st.session_state.green_state) < 0.25

=====================================================

MEMORY

=====================================================

def load_memory():
with open(MEM_FILE,"r",encoding="utf-8") as f:
return json.load(f)

def save_memory(M):
with open(MEM_FILE,"w",encoding="utf-8") as f:
json.dump(M,f,indent=2,ensure_ascii=False)

=====================================================

GITHUB AUTO SYNC

=====================================================

def github_sync():

try:  
    with open(MEM_FILE,"rb") as f:  
        content = base64.b64encode(f.read()).decode()  

    url=f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MEM_FILE}"  
    headers={"Authorization":f"token {GITHUB_TOKEN}"}  

    r=requests.get(url,headers=headers,timeout=10)  
    sha=r.json()["sha"] if r.status_code==200 else None  

    data={  
        "message":"🧬 Oracle memory auto-sync",  
        "content":content,  
        "branch":BRANCH  
    }  

    if sha:  
        data["sha"]=sha  

    requests.put(url,headers=headers,json=data,timeout=10)  

except Exception as e:  
    st.warning(f"Sync GitHub échoué : {e}")

=====================================================

Φ ENGINE

=====================================================

def evolve_phi(phi,exc):

phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))  
phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))  
phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))  

s=sum(phi.values())  
for k in phi:  
    phi[k]/=s  

return phi

=====================================================

HIPPOCAMPUS LEARNING

=====================================================

def learn(text,phi):

words=text.lower().split()  
if len(words)<2:  
    return  

energy=math.sqrt(sum(v*v for v in phi.values()))  
st.session_state.hippocampus.append((words,energy))  

if len(st.session_state.hippocampus)>5 and consolidation_gate():  
    consolidate()

def consolidate():

M=load_memory()  

for words,energy in st.session_state.hippocampus:  
    for a,b in zip(words,words[1:]):  
        M.setdefault(a,{})  
        M[a][b]=M[a].get(b,0)+energy  

save_memory(M)  
st.session_state.hippocampus.clear()  

github_sync()

=====================================================

SLEEP CYCLE

=====================================================

def sleep_cycle():

M=load_memory()  
new={}  

for w,con in M.items():  
    filt={t:v*0.997 for t,v in con.items() if v>1.2}  
    if filt:  
        new[w]=filt  

save_memory(new)  
st.session_state.last_sleep=time.time()

=====================================================

CORTEX

=====================================================

def contextual_seed(M):

ctx=" ".join(st.session_state.dialog).split()  
valid=[w for w in ctx if w in M]  

if valid:  
    return Counter(valid).most_common(1)[0][0]  

return random.choice(list(M.keys()))

def associative_layer(word,M,phi):

if word not in M:  
    return word  

opts=M[word]  

if random.random()<phi["phi_c"]:  
    return random.choices(  
        list(opts.keys()),  
        weights=list(opts.values())  
    )[0]  

return max(opts,key=opts.get)

def oracle_reply():

M=load_memory()  
if not M:  
    return "Mémoire vide."  

seed=contextual_seed(M)  
words=[seed]  

length=int(10+st.session_state.phi["phi_m"]*30)  

for _ in range(length):  
    nxt=associative_layer(words[-1],M,st.session_state.phi)  
    words.append(nxt)  

return " ".join(words).capitalize()+"."

=====================================================

FILE READING

=====================================================

def read_file(upload):

try:  
    if upload.type=="application/pdf":  
        reader=PyPDF2.PdfReader(upload)  
        return " ".join(p.extract_text() or "" for p in reader.pages)  

    if upload.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":  
        d=docx.Document(upload)  
        return " ".join(p.text for p in d.paragraphs)  

    if upload.type=="text/plain":  
        return upload.read().decode("utf-8",errors="ignore")  

    if upload.type=="text/csv":  
        return pd.read_csv(upload).to_string()  

except Exception as e:  
    st.error(f"Erreur lecture fichier : {e}")  

return ""

=====================================================

AUDIO INPUT

=====================================================

def speech_to_text(audio):

try:  
    r=sr.Recognizer()  
    with sr.AudioFile(audio) as source:  
        data=r.record(source)  
    return r.recognize_google(data)  
except:  
    return ""

=====================================================

UI

=====================================================

st.set_page_config(page_title="ORACLE V4.5 Ω",page_icon="🧠")
st.title("🧠 ORACLE V4.5 Ω — Agent Cognitif Total")

msg_input=st.text_input("Parlez à l'Oracle")

file=st.file_uploader(
"Insérer fichier / audio",
type=["pdf","docx","txt","csv","wav"]
)

=====================================================

PIPELINE COGNITIF

=====================================================

if st.button("Envoyer"):

progress=st.progress(0,text="🧠 Activation corticale...")  
status=st.empty()  

msg=""  

# PHASE 1 — INPUT  
if file:  
    status.info("📂 Lecture du fichier...")  
    progress.progress(20)  

    if file.type=="audio/wav":  
        msg=speech_to_text(file)  
    else:  
        msg=read_file(file)  
else:  
    msg=msg_input  

progress.progress(40)  

# PHASE 2 — ANALYSE  
status.info("🔎 Analyse cognitive...")  
time.sleep(0.2)  
progress.progress(60)  

# PHASE 3 — LEARNING  
if msg:  
    st.session_state.dialog.append(msg)  

    excitation=min(1,len(msg)/200)  
    st.session_state.phi=evolve_phi(  
        st.session_state.phi,excitation  
    )  

    learn(msg,st.session_state.phi)  

progress.progress(75)  

# PHASE 4 — RESPONSE  
status.info("💭 Génération pensée...")  
reply=oracle_reply()  
st.session_state.dialog.append(reply)  

progress.progress(90)  

# PHASE 5 — SYNC  
status.info("☁️ Synchronisation mémoire...")  
github_sync()  

progress.progress(100)  
status.success("✅ Oracle mis à jour")  
time.sleep(1)  
progress.empty()

=====================================================

CONVERSATION DISPLAY

=====================================================

for m in st.session_state.dialog:
st.write(m)

=====================================================

SIDEBAR

=====================================================

with st.sidebar:

st.header("🧠 État Cognitif")  

for k,v in st.session_state.phi.items():  
    st.progress(v,text=f"{k}:{v:.2f}")  

if st.button("🌙 Sommeil forcé"):  
    sleep_cycle()  
    st.success("Consolidation terminée")  

if st.button("🧬 Sync GitHub"):  
    github_sync()  
    st.success("Mémoire synchronisée")

Ne change ni les fonctionnalités ni la structure du code mais implémente le fantôme et les atouts de v5 voir v5.5 mais garde tout ajoute juste des qualités au bon endroit