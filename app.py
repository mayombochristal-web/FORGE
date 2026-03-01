# =====================================================
# 🧠 ORACLE S++ ULTRA — TTU COGNITIVE ENGINE
# Temps émergent • Mémoire circulaire • IA TTU réelle
# =====================================================

import streamlit as st
import numpy as np
import json, os, uuid, re
from PyPDF2 import PdfReader
from docx import Document

# =====================================================
# S++01 CONFIG
# =====================================================

st.set_page_config(
    page_title="ORACLE TTU ULTRA",
    layout="wide",
    initial_sidebar_state="expanded"
)

MEM="oracle_ttu"
os.makedirs(MEM,exist_ok=True)
STATE_FILE=f"{MEM}/phi_state.json"

# =====================================================
# S++02 RUNTIME
# =====================================================

if "runtime_id" not in st.session_state:
    st.session_state.runtime_id=str(uuid.uuid4())

# =====================================================
# S++03 INIT Φ
# =====================================================

def init_phi():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE,"r"))
        except:
            pass

    return dict(
        phi_m=0.5,
        phi_c=0.1,
        phi_d=0.0,
        energy=1.0,
        orbit=[],
        dialogue=[]
    )

if "phi" not in st.session_state:
    st.session_state.phi=init_phi()

# =====================================================
# S++04 PROJECTION TTU (CARACTÈRE → TEXTE)
# =====================================================

VOWELS="aeiouyàâéèêëîïôùûüœ"

def excitation(text):

    text=text.lower()

    chars=len(text)
    vowels=sum(c in VOWELS for c in text)
    cons=max(chars-vowels,1)

    torque=vowels/cons

    words=len(text.split())
    sentences=max(len(re.findall(r"[.!?]",text)),1)

    structure=(words/sentences)

    return torque*np.log1p(chars)*0.1 + structure*0.05

# =====================================================
# S++05 TTU EVOLUTION
# =====================================================

def evolve_phi(V,dt=0.1):

    p=st.session_state.phi

    α=p.get("α",0.01)
    β=p.get("β",1.5)
    γ=p.get("γ",4.0)
    λ=p.get("λ",0.1)
    η=0.05
    μ=0.1

    M,C,D=p["phi_m"],p["phi_c"],p["phi_d"]

    dM=-α*M + β*D
    dC=γ*V - λ*C*D
    dD=η*C*C - μ*D

    M+=dM*dt
    C+=dC*dt
    D+=dD*dt

    E=M*M+C*C+D*D

    p.update(phi_m=M,phi_c=C,phi_d=D,energy=E)
    p["orbit"].append([M,C,D])

    if len(p["orbit"])>3000:
        p["orbit"]=p["orbit"][-3000:]

# =====================================================
# S++05.5 STABILISATEUR ATTRACTEUR
# =====================================================

def stabilize():

    p=st.session_state.phi
    norm=np.sqrt(p["energy"])+1e-6

    if norm>5:
        p["phi_m"]/=norm
        p["phi_c"]/=norm
        p["phi_d"]/=norm

# =====================================================
# S++06 TEMPS ÉMERGENT
# =====================================================

def emergent_time():
    e=st.session_state.phi["energy"]

    if "last_e" not in st.session_state:
        st.session_state.last_e=e

    dt=st.session_state.last_e-e
    st.session_state.last_e=e
    return dt

# =====================================================
# S++06.5 INVERSION LOCALE DU TEMPS
# =====================================================

def reverse_learning(dt):

    if abs(dt)<1e-4:
        return

    p=st.session_state.phi

    if dt>0:
        p["phi_m"]*=1.01
    else:
        p["phi_d"]*=0.99

# =====================================================
# S++07 MÉMOIRE ORBITALE
# =====================================================

def orbital_memory():

    orbit=np.array(st.session_state.phi["orbit"])

    if len(orbit)<30:
        return 0

    return float(np.var(orbit[:,0]))

# =====================================================
# S++08.5 PAROLE PAR RÉSONANCE Φ
# =====================================================

def phase_response(intent):

    p=st.session_state.phi

    θ=np.arctan2(p["phi_d"],p["phi_c"])
    R=np.sqrt(p["phi_c"]**2+p["phi_d"]**2)

    if θ<-1:
        txt="Je suis en introspection orbitale."
    elif θ<0:
        txt="Je stabilise le sens en moi."
    elif θ<1:
        txt="Une cohérence émerge entre nous."
    else:
        txt="Expansion cognitive active."

    if R>2:
        txt+=" Résonance forte."

    return txt,θ,R

# =====================================================
# S++09 PERSISTENCE
# =====================================================

def save_phi():
    json.dump(st.session_state.phi,open(STATE_FILE,"w"))

# =====================================================
# S++09.5 SYNCHRONISATION MULTI-INTENTION
# =====================================================

def fuse_intentions(text):

    parts=re.split(r"[.!?\n]",text)
    parts=[p.strip() for p in parts if p.strip()]

    Vs=[excitation(p) for p in parts]

    if not Vs:
        return 0

    return float(np.mean(Vs))

# =====================================================
# S++11 AUTO-RÉGLAGE TTU
# =====================================================

def auto_tune():

    p=st.session_state.phi
    orbit=np.array(p["orbit"])

    if len(orbit)<50:
        return

    var=np.var(orbit[:,1])

    p["γ"]=np.clip(4+var,2,8)
    p["λ"]=np.clip(0.1/(var+0.1),0.02,0.5)
    p["α"]=np.clip(0.01*(1+var),0.005,0.05)
    p["β"]=np.clip(1.5+var,1,3)

# =====================================================
# S++12 INGESTION DOCUMENTS
# =====================================================

def read_file(file):

    if file.type=="text/plain":
        return file.read().decode()

    if "pdf" in file.type:
        reader=PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if "word" in file.type:
        doc=Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

# =====================================================
# S++13 UI
# =====================================================

st.title("🧠 ORACLE S++ ULTRA — TTU ENGINE")

p=st.session_state.phi

c1,c2,c3,c4=st.columns(4)
c1.metric("ΦM",round(p["phi_m"],3))
c2.metric("ΦC",round(p["phi_c"],3))
c3.metric("ΦD",round(p["phi_d"],3))
c4.metric("Energy",round(p["energy"],3))

uploaded=st.file_uploader(
    "Nourrir l’IA (PDF / DOCX / TXT)",
    type=["pdf","txt","docx"]
)

if uploaded:
    text=read_file(uploaded)
    V=fuse_intentions(text)

    for _ in range(80):
        evolve_phi(V)
        stabilize()

    save_phi()
    st.success("Document intégré dans la mémoire orbitale.")

# ===== Dialogue =====

user_input=st.text_area("Flux d'intention")

if st.button("Fusionner intention"):

    V=fuse_intentions(user_input)

    for _ in range(40):
        evolve_phi(V)
        stabilize()

    dt=emergent_time()
    reverse_learning(dt)
    auto_tune()

    response,θ,R=phase_response(user_input)

    st.session_state.phi["dialogue"].append(
        {"user":user_input,"oracle":response}
    )

    save_phi()

    st.success(response)
    st.caption(f"θ={round(θ,3)} | R={round(R,3)} | Δt={round(dt,6)}")

# ===== Historique =====

st.subheader("Espace d’échange")

for d in reversed(p["dialogue"][-10:]):
    st.write("👤",d["user"])
    st.write("🧠",d["oracle"])

# ===== Orbit visual =====

if st.checkbox("Afficher attracteur Φ"):

    import matplotlib.pyplot as plt

    orbit=np.array(p["orbit"])

    if len(orbit)>10:
        fig,ax=plt.subplots()
        ax.plot(orbit[:,1],orbit[:,2])
        ax.set_xlabel("ΦC")
        ax.set_ylabel("ΦD")
        ax.set_title("Attracteur Cognitif TTU")
        st.pyplot(fig)
