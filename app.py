import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
from engine import TTUEngine
from demodulator import Demodulator
from manager import DBManager

st.set_page_config(page_title="TTU-MC3 AI", layout="wide")

if 'ttu' not in st.session_state:
    st.session_state.ttu = TTUEngine()
    st.session_state.demod = Demodulator()
    st.session_state.db = DBManager()
    st.session_state.chat = []

st.title("🌌 TTU-MC³ Generator")
st.caption("Mode : Dissipation Complexe Active")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

prompt = st.chat_input("Envoyez une impulsion...")

if prompt:
    st.session_state.chat.append({"role": "user", "text": prompt})
    
    # Injection d'impulsion et Jitter de Phase
    imp_base = sum(ord(c) for c in prompt) / 500.0
    m_history, full_states = [], []
    
    for i in range(120): # Cycles étendus pour plus de substance
        # Choc de phase aléatoire pour briser la linéarité
        jitter = random.uniform(-0.1, 0.1) if i % 5 == 0 else 0
        s = st.session_state.ttu.process_signal(imp_base + jitter)
        m_history.append(s[2])
        full_states.append(s)
    
    # Vérification SGBD
    sig = str(round(sum(m_history[:3]), 2))
    mem = st.session_state.db.check_memory(sig)
    
    if mem:
        response = mem[0]
    else:
        response = st.session_state.demod.decode_stream(m_history)
        st.session_state.db.save_crystal(sig, response)
    
    st.session_state.chat.append({"role": "assistant", "text": response})
    st.session_state.last_data = np.array(full_states)
    st.rerun()

if 'last_data' in st.session_state:
    st.divider()
    data = st.session_state.last_data
    fig = go.Figure(data=[go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2], 
                    mode='lines', line=dict(color='cyan', width=4))])
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
