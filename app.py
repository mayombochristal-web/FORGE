# =====================================================
# 🧠 ORACLE APP — INTERFACE SENSORIELLE
# =====================================================

import streamlit as st
import time
import PyPDF2, docx

import oracle_core as brain

# =====================================================
# SESSION STATE = ORGANISME
# =====================================================

if "brain_state" not in st.session_state:
    st.session_state.brain_state={
        "phi":{"phi_m":0.5,"phi_c":0.5,"phi_d":0.5},
        "green_state":0.0,
        "ghost_cache":[],
        "identity_entropy":0.5,
        "last_sleep":time.time()
    }

if "dialog_memory" not in st.session_state:
    st.session_state.dialog_memory=[]

state=st.session_state.brain_state

brain.auto_sleep(state)

# =====================================================
# FILE READER
# =====================================================

def read_file(upload):

    text=""

    if upload.name.endswith(".pdf"):
        reader=PyPDF2.PdfReader(upload)
        for p in reader.pages:
            text+=p.extract_text() or ""

    elif upload.name.endswith(".docx"):
        doc=docx.Document(upload)
        text="\n".join(p.text for p in doc.paragraphs)

    return text

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V3.2")

st.title("🧠 ORACLE — Cognitive System")

msg=st.text_input("Parlez à l'Oracle")
uploaded=st.file_uploader("Insérer document")

if uploaded:
    txt=read_file(uploaded)
    brain.learn(state,txt)

if st.button("➡️") and msg:

    st.session_state.dialog_memory.append(msg)

    brain.process_input(state,msg)

    reply=brain.oracle_reply(state)

    st.session_state.dialog_memory.append(reply)

    brain.learn(state,reply,0.3)

for m in st.session_state.dialog_memory:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("État Cognitif")

    for k,v in state["phi"].items():
        st.progress(v,text=f"{k}: {v:.2f}")

    st.write(
        f"Identity entropy : {state['identity_entropy']:.2f}"
    )

    if st.button("🌙 Sommeil forcé"):
        removed=brain.sleep_cycle(state)
        st.warning(f"{removed} synapses oubliées")
        st.rerun()