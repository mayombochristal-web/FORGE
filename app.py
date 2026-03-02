# =====================================================
# 🧠 ORACLE V6 — CORTEX STREAMLIT
# =====================================================

import streamlit as st
from collections import deque
from io import BytesIO

import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr

import oracle_core as core

# =====================================================
# SESSION UI
# =====================================================

if "dialog" not in st.session_state:
    st.session_state.dialog=deque(maxlen=60)

# =====================================================
# FILE READER SAFE
# =====================================================

def read_file(upload):

    raw=upload.read()

    try:
        if upload.type=="application/pdf":
            reader=PyPDF2.PdfReader(BytesIO(raw))
            return " ".join(
                p.extract_text() or ""
                for p in reader.pages[:40]
            )

        if upload.type.endswith("document"):
            doc=docx.Document(BytesIO(raw))
            return " ".join(p.text for p in doc.paragraphs[:400])

        if upload.type=="text/plain":
            return raw.decode("utf-8","ignore")

        if upload.type=="text/csv":
            df=pd.read_csv(BytesIO(raw))
            return df.head(400).to_string()

    except:
        return ""

    return ""

# =====================================================
# AUDIO
# =====================================================

def speech_to_text(file):
    try:
        r=sr.Recognizer()
        with sr.AudioFile(file) as src:
            audio=r.record(src)
        return r.recognize_google(audio)
    except:
        return ""

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="ORACLE V6",page_icon="🧠")

st.title("🧠 ORACLE V6 — Cortex Interface")

msg_input=st.text_input("Parlez à l'Oracle")

file=st.file_uploader(
    "Insérer fichier / audio",
    type=["pdf","docx","txt","csv","wav"]
)

if st.button("Envoyer"):

    msg=""

    if file:
        if file.type=="audio/wav":
            msg=speech_to_text(file)
        else:
            msg=read_file(file)
    else:
        msg=msg_input

    if msg:

        core.ghost_preload(msg)

        st.session_state.dialog.append(msg)

        exc=min(1,len(msg)/200)
        core.evolve_phi(exc)

        core.learn(msg)

        reply=core.generate(st.session_state.dialog)

        st.session_state.dialog.append(reply)

        core.learn(reply,0.3)

# =====================================================
# DISPLAY
# =====================================================

for m in st.session_state.dialog:
    st.write(m)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    for k,v in core.brain_state["phi"].items():
        st.progress(v,text=f"{k}: {v:.2f}")

    if st.button("🌙 Sommeil forcé"):
        core.sleep_cycle()
        st.success("Consolidation terminée")