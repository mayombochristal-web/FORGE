# =====================================================
# 🧠 ORACLE V6 Ω — BIOLOGICAL WORKSPACE
# =====================================================

import streamlit as st
from oracle_core import *

st.set_page_config(page_title="ORACLE V6 Ω", layout="wide")

st.title("🧠 ORACLE V6 Ω — Biological Persistent Cortex")

memory = load_memory()

# =====================================================
# GLOBAL WORKSPACE DISPLAY
# =====================================================

for msg in memory["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================
# INPUT ZONE
# =====================================================

col1,col2=st.columns([4,1])

with col1:
    prompt = st.chat_input("Message au cortex...")

with col2:
    file = st.file_uploader("Insert/compiler")

# =====================================================
# PROCESS
# =====================================================

if prompt or file:

    text=""

    if file:
        text=read_file(file)

    if prompt:
        text+=(" "+prompt)

    if text.strip():
        process_input(text)

    st.rerun()

# =====================================================
# AUTO SYNC
# =====================================================

auto_sync_loop()

# =====================================================
# STATUS BAR (HOMEOSTASIS)
# =====================================================

st.divider()

c1,c2,c3=st.columns(3)

with c1:
    st.caption(
        "🧠 Φ : " +
        ", ".join(f"{k}:{round(v,2)}"
        for k,v in st.session_state.phi.items())
    )

with c2:
    state="🟡 pending" if st.session_state.memory_dirty else "🟢 synced"
    st.caption(f"Memory {state}")

with c3:
    st.caption(f"Green noise: {round(st.session_state.green_state,3)}")