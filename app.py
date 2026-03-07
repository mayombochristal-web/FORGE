import streamlit as st
from oracle_engine_v13 import OracleBrain
import time

st.set_page_config(
    page_title="ORACLE Ω-TTU V13 OMEGA",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 ORACLE Ω-TTU V13 OMEGA")
st.caption("Agent Cognitif Dynamique — TTU-MC³ + Mémoire Vectorielle")

# =====================================================
# INITIALISATION ORACLE
# =====================================================

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain()

oracle = st.session_state.oracle

# =====================================================
# SIDEBAR : ETAT COGNITIF
# =====================================================

with st.sidebar:

    st.header("État Cognitif")

    phi = oracle.phi

    st.metric("Φ Mémoire", f"{phi['phi_m']:.3f}")
    st.progress(phi["phi_m"])

    st.metric("Φ Cohérence", f"{phi['phi_c']:.3f}")
    st.progress(phi["phi_c"])

    st.metric("Φ Dissipation", f"{phi['phi_d']:.3f}")
    st.progress(phi["phi_d"])

    st.divider()

    st.metric("Distance Attracteur", f"{oracle.distance():.4f}")

    st.metric("Concepts mémoire", oracle.memory_size())

    st.metric("Age cognition", oracle.age)

    st.divider()

    if st.button("🌙 Cycle de Sommeil"):
        oracle.sleep_cycle()
        st.success("Mémoire consolidée")

# =====================================================
# INTERFACE DIALOGUE
# =====================================================

user = st.text_input("Parlez à l'Oracle")

if user:

    oracle.learn(user)

    response = oracle.think()

    st.write("### Oracle")
    st.write(response)

    oracle.feedback(response, True)

# =====================================================
# VISUALISATION
# =====================================================

if st.button("Carte Conceptuelle"):

    fig = oracle.visualize()

    if fig:
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# AUTO REFRESH
# =====================================================

time.sleep(0.2)