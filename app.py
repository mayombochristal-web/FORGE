import streamlit as st
import random

from core.phi_engine import *
from core.memory_engine import *
from core.language_engine import *
from core.ingestion_engine import *
from core.sleep_engine import *

# =====================================
# S+06 CONFIG
# =====================================

st.set_page_config(
    page_title="ORACLE S+ AUTONOME",
    page_icon="🧠"
)

if "phi" not in st.session_state:
    st.session_state.phi=init_phi()

# =====================================
# UI
# =====================================

st.title("🧠 ORACLE S+ — Proto Cognition")

st.sidebar.header("État Φ")
st.sidebar.json(st.session_state.phi)

# =====================================
# INGESTION
# =====================================

mode=st.radio(
    "Source",
    ["Message","PDF","Audio WAV"]
)

user_text=None

if mode=="Message":
    user_text=st.text_input("Parlez à ORACLE")

elif mode=="PDF":
    f=st.file_uploader("PDF",type="pdf")
    if f:
        user_text=read_pdf(f)

elif mode=="Audio WAV":
    a=st.file_uploader("Audio",type="wav")
    if a:
        user_text=read_audio(a)
        st.info(user_text)

# =====================================
# INTERACTION
# =====================================

if st.button("Émettre / Apprendre") and user_text:

    phi=st.session_state.phi

    phi=evolve_phi(phi,0.3)

    intensity=phi_intensity(phi)

    learn(user_text,intensity)

    reply=oracle_reply(phi)

    st.markdown(f"### 💬 {reply}")

    if random.random()<0.2:
        thought=oracle_reply(phi)
        learn(thought,0.4)
        st.sidebar.info(f"Pensée interne : {thought}")

# =====================================
# SYSTEM TOOLS
# =====================================

st.divider()

c1,c2=st.columns(2)

with c1:
    if st.button("🌙 Sommeil"):
        st.success(sleep_cycle())

with c2:
    if st.button("🗑 Reset mémoire"):
        save_lex({})
        st.warning("Mémoire effacée.")