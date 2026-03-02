# =====================================================
# 🧠 ORACLE V6.1 — AUTONOMOUS CORTEX APP
# =====================================================

import streamlit as st
from oracle_core import *

st.set_page_config(page_title="ORACLE V6.1", layout="wide")

# INIT SESSION
if "memory_dirty" not in st.session_state:
    st.session_state.memory_dirty = False

# =====================================================
# LOAD MEMORY
# =====================================================

memory = load_memory()

st.title("🧠 ORACLE V6.1 — Auto Persistent Cortex")

# =====================================================
# DISPLAY CHAT
# =====================================================

for msg in memory["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================
# USER INPUT
# =====================================================

prompt = st.chat_input("Message...")

if prompt:

    add_message("user", prompt)

    response = f"Oracle processed: {prompt}"

    add_message("assistant", response)

    st.rerun()

# =====================================================
# AUTO SYNC (NO BUTTON)
# =====================================================

auto_sync_loop()

# =====================================================
# STATUS
# =====================================================

if st.session_state.memory_dirty:
    st.caption("🟡 Memory pending sync...")
else:
    st.caption("🟢 Memory synced")
st.write("Repo:", st.secrets["GITHUB_REPO"])