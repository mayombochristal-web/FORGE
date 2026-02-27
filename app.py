import streamlit as st
from oracle_agent import OracleAgent
from openai import OpenAI
import time

st.set_page_config(layout="wide")

st.title("🔮 TTU ORACLE VIVANT — V4")

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleAgent()

oracle = st.session_state.oracle

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------- INPUT UTILISATEUR ----------
user_input = st.chat_input("Parler à l'Oracle...")

# ---------- RÉPONSE UTILISATEUR ----------
if user_input:

    oracle.update_memory("user", user_input)

    messages = [
        {"role": "system", "content": oracle.autonomous_thought()},
        *oracle.memory.context(),
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content

    oracle.update_memory("assistant", reply)
    oracle.evaluate(reply)

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("assistant", reply))

# ---------- MODE AUTONOME ----------
if st.sidebar.button("🧠 Pensée autonome"):

    thought = oracle.autonomous_thought()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": thought}]
    )

    auto_reply = response.choices[0].message.content

    oracle.update_memory("assistant", auto_reply)
    oracle.evaluate(auto_reply)

    st.session_state.chat.append(("assistant", auto_reply))

# ---------- AFFICHAGE ----------
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

st.sidebar.metric("Vitalité Spectrale VS", round(oracle.vs,2))
st.sidebar.write("Identité:", oracle.identity)
