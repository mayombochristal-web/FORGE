# =====================================================
# 🧠 ORACLE V6 Ω — Biological Persistent Cortex UI
# =====================================================

import streamlit as st
from oracle_core import *

st.set_page_config(page_title="ORACLE V6 Ω", layout="wide")

st.title("🧠 ORACLE V6 Ω — Biological Persistent Cortex")

memory = load_memory()

# =====================================================
# CHAT DISPLAY
# =====================================================

for msg in memory["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================
# INPUT
# =====================================================

uploaded = st.file_uploader(
    "Insert file / audio",
    type=["pdf","docx","txt","csv","wav"]
)

prompt = st.chat_input("Message...")

if prompt or uploaded:

    if uploaded:
        if uploaded.type=="audio/wav":
            text=speech_to_text(uploaded)
        else:
            text=read_file(uploaded)
    else:
        text=prompt

    if text:
        process_input(text)
        st.rerun()

# =====================================================
# STATUS BAR
# =====================================================

with st.sidebar:

    st.header("🧠 Cognitive State")

    for k,v in st.session_state.phi.items():
        st.progress(v,text=f"{k}: {v:.2f}")

    if st.button("🌙 Sleep now"):
        sleep_cycle()
        st.success("Consolidation complete")