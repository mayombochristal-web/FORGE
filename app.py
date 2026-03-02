# =====================================================
# 🧠 ORACLE APP — V3.2 Ω INTERFACE
# =====================================================

import streamlit as st
from oracle_core import process_input, brain

st.set_page_config(page_title="Oracle V3.2 Ω", layout="wide")

st.title("🧠 ORACLE V3.2 Ω — Cognitive System")

# session state
if "history" not in st.session_state:
    st.session_state.history = []

# input
user_input = st.chat_input("Parle à l'Oracle...")

if user_input:

    response = process_input(user_input)

    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("oracle", response))

# display chat
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# debug panel
with st.sidebar:
    st.header("🧬 Brain State")
    st.write("Φ:", round(brain.phi, 3))
    st.write("Dialog size:", len(brain.dialog_memory))
    st.write("Long memory:", len(brain.long_memory))