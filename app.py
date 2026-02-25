import streamlit as st
import pandas as pd
import numpy as np
from core.engine import TTUEngine
from interface.demodulator import Demodulator
from db.manager import DBManager
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="TTU-MC³ Generator", layout="wide")
st.title("🧠 TTU-MC³ : Générateur par Dissipation")

# Initialisation des composants dans la session Streamlit
if 'engine' not in st.session_state:
    st.session_state.engine = TTUEngine()
    st.session_state.demod = Demodulator()
    st.session_state.db = DBManager()
    st.session_state.history = []

# Sidebar pour le monitoring interne
st.sidebar.header("🎛️ Monitoring de Phase")
gain_val = st.sidebar.slider("Gain (γ)", 0.5, 2.0, 1.1)
st.session_state.engine.gamma = gain_val

# Zone de Chat
st.subheader("Conversation avec la Substance")
container = st.container()

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Entrez votre prompt :", placeholder="L'IA va cristalliser votre pensée...")
    submit_button = st.form_submit_button(label='Dissiper & Générer')

if submit_button and user_input:
    # 1. Calcul de l'impulsion
    impulse = sum(ord(c) for c in user_input) / 1000.0
    
    # 2. Simulation et capture des données pour le monitoring
    phase_data = []
    for _ in range(60):
        state = st.session_state.engine.process_signal(impulse)
        phase_data.append({"Phi_C": state[0], "Phi_D": state[1], "Phi_M": state[2]})
    
    df_phase = pd.DataFrame(phase_data)
    
    # 3. Démodulation
    response = st.session_state.demod.get_ascii(df_phase['Phi_M'].tolist())
    
    # 4. Archivage
    st.session_state.history.append({"user": user_input, "ai": response, "data": df_phase})

# Affichage des résultats
if st.session_state.history:
    last_entry = st.session_state.history[-1]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"**Dernière réponse :** {last_entry['ai']}")
        # Graphique de l'Attracteur
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=last_entry['data']['Phi_M'], mode='lines', name='Mémoire (Φm)', line=dict(color='gold')))
        fig.add_trace(go.Scatter(y=last_entry['data']['Phi_D'], mode='lines', name='Dissipation (Φd)', line=dict(color='red')))
        fig.update_layout(title="Dynamique de l'Attracteur", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Espace de Phase (Φc vs Φd)**")
        fig_phase = go.Figure(data=go.Scatter(x=last_entry['data']['Phi_C'], y=last_entry['data']['Phi_D'], mode='markers+lines'))
        fig_phase.update_layout(xaxis_title="Cohérence", yaxis_title="Dissipation", template="plotly_dark")
        st.plotly_chart(fig_phase, use_container_width=True)
