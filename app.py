# =====================================================
# 🧠 ORACLE V6 Ω — CORTEX SENSORIEL
# =====================================================

import streamlit as st
from oracle_core import *

st.set_page_config(page_title="ORACLE V6 Ω",layout="wide")

st.title("🧠 ORACLE V6 Ω — Cortex Autonome")

memory=load_memory()

# =====================================================
# DISPLAY CHAT
# =====================================================

for msg in memory["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================
# INPUT
# =====================================================

prompt=st.chat_input("Message à l'Oracle")

file=st.file_uploader(
    "Insert fichier / audio",
    type=["pdf","docx","txt","csv","wav"]
)

if prompt or file:

    text=""

    if file:
        if file.type=="audio/wav":
            text=speech_to_text(file)
        else:
            text=read_file(file)
    else:
        text=prompt

    if text:
        process_input(text)
        st.rerun()

# =====================================================
# AUTO SYNC
# =====================================================

auto_sync_loop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧠 État Cognitif")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}:{v:.2f}")

    if st.button("🌙 Sommeil forcé"):
        sleep_cycle()
        st.rerun()

    if st.session_state.memory_dirty:
        st.caption("🟡 Sync en attente")
    else:
        st.caption("🟢 Mémoire synchronisée")