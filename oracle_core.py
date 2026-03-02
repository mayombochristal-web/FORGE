# =====================================================
# 🧠 ORACLE CORE V3.2 CANONICAL
# Fusion V3.1 + V4.5 Ω + V6 Ω
# =====================================================

import random, json, os, math, time
from collections import deque, Counter
from io import BytesIO

import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr

# ================= CONFIG =================

MEM_DIR="oracle_memory"
LEXICON_PATH=os.path.join(MEM_DIR,"lexicon.json")

os.makedirs(MEM_DIR,exist_ok=True)

if not os.path.exists(LEXICON_PATH):
    json.dump({},open(LEXICON_PATH,"w",encoding="utf-8"))

MAX_FILE_MB=8
MAX_PAGES=40
MAX_ROWS=400

# ================= BRAIN STATE =================

brain={
    "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
    "dialog_memory":deque(maxlen=60),
    "green_state":0.0,
    "last_sleep":time.time(),
    "ghost_cache":{},
    "hippocampus":[],
    "identity_entropy":0.5
}

# ================= MEMORY =================

def load_lex():
    try:
        return json.load(open(LEXICON_PATH,"r",encoding="utf-8"))
    except:
        return {}

def save_lex(L):

    if len(L)>15000:
        L=dict(list(L.items())[-12000:])

    json.dump(
        L,
        open(LEXICON_PATH,"w",encoding="utf-8"),
        indent=2,
        ensure_ascii=False
    )

# ================= GREEN NOISE =================

def green_noise(prev):
    return 0.92*prev+0.08*random.uniform(-1,1)

def consolidation_gate():
    brain["green_state"]=green_noise(brain["green_state"])
    return abs(brain["green_state"])<0.25

# ================= Φ ENGINE =================

def evolve_phi(exc):

    phi=brain["phi"]

    phi["phi_m"]=min(1,max(0.1,phi["phi_m"]+exc*0.15-0.01))
    phi["phi_c"]=min(1,max(0.1,phi["phi_c"]+exc*0.3-0.03))
    phi["phi_d"]=min(1,max(0.1,phi["phi_d"]+0.02-exc*0.05))

    # influence biologique green noise
    phi["phi_c"]+=brain["green_state"]*0.05

    s=sum(phi.values())
    for k in phi:
        phi[k]/=s

# ================= GHOST CORTEX =================

def ghost_preload(text):

    L=load_lex()
    cache={}

    for w in text.lower().split():
        if w in L:
            cache[w]=sorted(
                L[w].items(),
                key=lambda x:-x[1]
            )[:5]

    brain["ghost_cache"]=cache

# ================= HIPPOCAMPUS =================

def learn(text,importance=1.0):

    words=text.lower().split()
    if len(words)<2:
        return

    energy=math.sqrt(sum(v*v for v in brain["phi"].values()))*importance
    brain["hippocampus"].append((words,energy))

    if len(brain["hippocampus"])>5 and consolidation_gate():
        consolidate()

def consolidate():

    L=load_lex()

    for words,energy in brain["hippocampus"]:
        for a,b in zip(words,words[1:]):
            L.setdefault(a,{})
            L[a][b]=L[a].get(b,0)+energy

    if len(L)>12000:
        L=dict(list(L.items())[-10000:])

    save_lex(L)
    brain["hippocampus"].clear()

# ================= SLEEP =================

def sleep_cycle():

    L=load_lex()
    new={}

    for w,con in L.items():
        filt={t:v*0.997 for t,v in con.items() if v>1.2}
        if filt:
            new[w]=filt

    save_lex(new)
    brain["last_sleep"]=time.time()

def auto_sleep():
    if time.time()-brain["last_sleep"]>180:
        if consolidation_gate():
            sleep_cycle()

# ================= SENSORIUM =================

def extract_text_from_file(upload):

    raw=upload.read()

    if len(raw)>MAX_FILE_MB*1024*1024:
        return ""

    try:

        if upload.type=="application/pdf":
            reader=PyPDF2.PdfReader(BytesIO(raw))
            text=[]
            for p in reader.pages[:MAX_PAGES]:
                t=p.extract_text()
                if t:
                    text.append(t)
            return " ".join(text)

        if upload.type.endswith("document"):
            doc=docx.Document(BytesIO(raw))
            return " ".join(p.text for p in doc.paragraphs[:400])

        if upload.type=="text/plain":
            return raw.decode("utf-8","ignore")

        if upload.type=="text/csv":
            df=pd.read_csv(BytesIO(raw))
            return df.head(MAX_ROWS).to_string()

    except:
        pass

    return ""

def speech_to_text(file):
    try:
        r=sr.Recognizer()
        with sr.AudioFile(file) as src:
            audio=r.record(src)
        return r.recognize_google(audio)
    except:
        return ""

# ================= THALAMUS =================

def contextual_seed(L):

    ctx=" ".join(brain["dialog_memory"]).split()
    valid=[w for w in ctx if w in L]

    if valid:
        return Counter(valid).most_common(1)[0][0]

    return random.choice(list(L.keys()))

# ================= CORTEX =================

def associative_layer(word,L):

    ghost=brain["ghost_cache"].get(word)
    if ghost:
        return random.choice(ghost)[0]

    if word not in L:
        return word

    opts=L[word]

    if random.random()<brain["phi"]["phi_c"]:
        return random.choices(
            list(opts.keys()),
            weights=list(opts.values())
        )[0]

    return max(opts,key=opts.get)

# ================= GENERATION =================

def generate():

    auto_sleep()

    L=load_lex()
    if not L:
        return "Mémoire vide."

    seed=contextual_seed(L)

    words=[seed]
    used=set(words)

    length=int(10+brain["phi"]["phi_m"]*30)

    for _ in range(length):

        nxt=associative_layer(words[-1],L)

        if nxt in used and random.random()<brain["phi"]["phi_d"]:
            break

        words.append(nxt)
        used.add(nxt)

    if brain["phi"]["phi_c"]>0.65:
        words.append("évolution")

    return " ".join(words).capitalize()+"."

# ================= IDENTITY =================

def evolve_identity():
    brain["identity_entropy"]+=random.uniform(-0.01,0.01)

# ================= MASTER PIPELINE =================

def process_input(text=None,file=None):

    # perception
    if file:
        if file.type=="audio/wav":
            text=speech_to_text(file)
        else:
            text=extract_text_from_file(file)

    if not text:
        return "Aucune information perçue."

    evolve_identity()

    ghost_preload(text)

    brain["dialog_memory"].append(text)

    ghost_factor=len(brain["ghost_cache"])/5
    exc=min(1,(len(text)/200)+ghost_factor*0.2)

    evolve_phi(exc)

    learn(text,1.2)

    reply=generate()

    brain["dialog_memory"].append(reply)

    learn(reply,0.3)

    return reply