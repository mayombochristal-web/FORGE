# =====================================================
# 🧠 ORACLE V6.1 — AUTONOMOUS CORTEX APP
# Streamlit = Cortex Sensoriel
# oracle_core = Cerveau vivant
# =====================================================

import streamlit as st

# =====================================================
# IMPORT ORACLE CORE
# =====================================================

from oracle_core import (
    load_memory,
    add_message,
    auto_sync_loop
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="ORACLE V6.1",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# SESSION INIT (ANTI RERUN BUG)
# =====================================================

if "memory_dirty" not in st.session_state:
    st.session_state.memory_dirty = False

if "initialized" not in st.session_state:
    st.session_state.initialized = True

# =====================================================
# SAFE MEMORY LOAD
# =====================================================

try:
    memory = load_memory()
except Exception as e:
    st.error("❌ Memory loading failed")
    st.stop()

# sécurité structure
if "messages" not in memory:
    memory["messages"] = []

# =====================================================
# UI HEADER
# =====================================================

st.title("🧠 ORACLE V6.1 — Auto Persistent Cortex")

# =====================================================
# CHAT DISPLAY
# =====================================================

chat_container = st.container()

with chat_container:

    for msg in memory["messages"]:

        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        with st.chat_message(role):
            st.write(content)

# =====================================================
# USER INPUT
# =====================================================

prompt = st.chat_input("Message...")

if prompt:

    # USER MESSAGE
    add_message("user", prompt)

    # ORACLE RESPONSE (placeholder cortex)
    response = f"Oracle processed: {prompt}"

    add_message("assistant", response)

    # rerun propre
    st.rerun()

# =====================================================
# AUTO SYNC BACKGROUND
# =====================================================

try:
    auto_sync_loop()
except Exception:
    # jamais bloquer UI
    pass

# =====================================================
# STATUS BAR
# =====================================================

st.divider()

col1, col2 = st.columns([1,1])

with col1:
    if st.session_state.memory_dirty:
        st.caption("🟡 Memory pending sync...")
    else:
        st.caption("🟢 Memory synced")

with col2:
    try:
        repo = st.secrets["GITHUB_REPO"]
        st.caption(f"Repo: {repo}")
    except Exception:
        st.caption("⚠️ GitHub repo not configured")

# =====================================================
# FOOTER (DEBUG SAFE)
# =====================================================

with st.expander("⚙️ Cortex Status", expanded=False):

    st.write("Messages stored:", len(memory["messages"]))
    st.write("Session active:", st.session_state.initialized)