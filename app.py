import streamlit as st
from oracle_core import *

# IMPORTANT — INITIALISATION
init_state()

st.title("🧠 ORACLE V6 Ω — Biological Cortex")

mem = load_memory()

# afficher phi
st.subheader("Φ State")
for k,v in st.session_state.phi.items():
    st.write(f"{k}: {round(v,3)}")

# historique
for m in mem["messages"][-10:]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# input
prompt = st.chat_input("Parle à Oracle...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    reply = process_input(prompt)

    with st.chat_message("assistant"):
        st.write(reply)