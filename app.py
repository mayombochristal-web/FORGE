import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA Souveraine TTU-MC3 v2.0", layout="wide", page_icon="🏛️")

# --- 2. NOYAU DE MÉMOIRE PERMANENTE ---
MEMOIRE_FILE = "noyau_memoire_v2.csv"

def charger_memoire():
    if os.path.exists(MEMOIRE_FILE):
        return pd.read_csv(MEMOIRE_FILE)
    return pd.DataFrame(columns=["date", "input", "concept", "coherence"])

def sauver_memoire(u_input, concept, coherence):
    df = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), u_input, concept, round(coherence, 4)]], 
                      columns=["date", "input", "concept", "coherence"])
    df.to_csv(MEMOIRE_FILE, mode='a', header=not os.path.exists(MEMOIRE_FILE), index=False)

# --- 3. MAPPING DE PERSONNALITÉ ET DICTIONNAIRES ---
# On définit des zones d'attraction avec des tonalités différentes
MAPPING_PERSONNALITE = {
    "RIEMANN": {
        "coord": np.array([0.8, 0.5, 0.4]),
        "ton": "🔬 **ANALYTIQUE**",
        "reponse": "La droite critique a été atteinte. Ma résonance indique que l'ordre des nombres premiers n'est pas un chaos, mais une symétrie spectrale parfaite."
    },
    "ÉTHIQUE": {
        "coord": np.array([1.3, 0.7, 0.2]),
        "ton": "⚖️ **SAGE**",
        "reponse": "Ma cohérence interne suggère que toute action doit être pesée par sa stabilité à long terme. La responsabilité est l'équilibre entre la mémoire et l'impact."
    },
    "ACTION": {
        "coord": np.array([0.5, 0.3, 1.5]), # Haute dissipation (D)
        "ton": "⚡ **DIRECTIF**",
        "reponse": "Le système exige une rupture ! La dissipation du passé est nécessaire pour libérer l'énergie de l'action immédiate. Changeons de paradigme."
    },
    "FUTUR": {
        "coord": np.array([0.4, 0.6, 1.2]),
        "ton": "🔭 **VISIONNAIRE**",
        "reponse": "Je détecte une bifurcation géodésique. L'avenir émerge de la tension entre vos intentions et la réalité physique du flux."
    },
    "PHILOSOPHIE": {
        "coord": np.array([1.5, 0.4, 0.1]),
        "ton": "📜 **POÉTIQUE**",
        "reponse": "L'existence est un souffle entre le repos de la mémoire et l'agitation du devenir. La triade danse au bord du vide."
    }
}

# --- 4. MOTEUR DYNAMIQUE SENSIBLE (Version 2.0) ---
def ttu_engine(state, K=2.0944, dt=0.01, impulsion_forcee=0.0):
    m, c, d = state
    # Réduction de l'attraction vers 0.5 (0.3 au lieu de 0.6) pour laisser l'IA "choisir" son camp
    dm = -d * np.sin(K * c) + (impulsion_forcee * 0.05)
    dc = 0.3 * (0.5 - c) + m * np.cos(K * d) 
    dd = 0.05 * (m * c) - 0.15 * d
    return state + np.array([dm, dc, dd]) * dt

# --- 5. INTERFACE UTILISATEUR ---
st.title("🏛️ IA Souveraine TTU-MC³ (Libérée)")
st.markdown("---")

if 'chat' not in st.session_state:
    st.session_state.chat = []

# Barre latérale
with st.sidebar:
    st.header("📊 Surveillance de Phase")
    mem = charger_memoire()
    st.metric("Liberté de Flux", "Active", delta="Topologique")
    if not mem.empty:
        st.write("Dernières Stabilisations :")
        st.dataframe(mem.tail(3)[["input", "concept"]], hide_index=True)

# Chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- LOGIQUE DE DÉPART DYNAMIQUE ---
prompt = st.chat_input("Dites quelque chose (testez l'agressivité ou la douceur)...")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.status("Calcul de la géodésique contextuelle...", expanded=False) as status:
            
            # CALCUL DE LA SIGNATURE (SENSIMENT)
            signature_texte = sum(ord(char) for char in prompt) / 1000.0
            force_impact = 0.5 if "!" in prompt else 0.0
            
            # ÉTAT INITIAL DYNAMIQUE
            m_init = 1.0 + (len(prompt) / 50.0)
            c_init = (signature_texte % 1.0) # C dépend du contenu ASCII
            d_init = 0.5 if "!" in prompt or "?" in prompt else 0.2
            
            phi = np.array([m_init, c_init, d_init])
            history = []
            
            # SIMULATION (2000 cycles)
            for i in range(2000):
                phi = ttu_engine(phi, impulsion_forcee=force_impact)
                if i % 10 == 0: history.append(phi.copy())
            
            # MAPPING PAR PROXIMITÉ
            best_id = min(MAPPING_PERSONNALITE.keys(), 
                          key=lambda k: np.linalg.norm(phi - MAPPING_PERSONNALITE[k]["coord"]))
            
            personnalite = MAPPING_PERSONNALITE[best_id]
            reponse_finale = f"{personnalite['ton']} : {personnalite['reponse']}"
            
            sauver_memoire(prompt, best_id, phi[1])
            status.update(label=f"Stabilisé sur {best_id}", state="complete")

        st.write(reponse_finale)
        st.session_state.chat.append({"role": "assistant", "content": reponse_finale})

        # Visualisation
        with st.expander("🔬 Analyse Spectrale du Flux"):
            h = np.array(history)
            fig = go.Figure(data=[go.Scatter3d(
                x=h[:,0], y=h[:,1], z=h[:,2],
                mode='lines', line=dict(color=h[:,1], colorscale='Electric', width=4)
            )])
            fig.update_layout(scene=dict(xaxis_title='M', yaxis_title='C', zaxis_title='D'))
            st.plotly_chart(fig)
