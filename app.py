import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
from scipy.fft import fft, fftfreq

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA TTU-MC3 Spectral Ghost", layout="wide", page_icon="👻")

# --- 2. GESTION DU NOYAU DE MÉMOIRE ---
MEMOIRE_FILE = "noyau_souverain_spectral.csv"

def charger_memoire():
    if os.path.exists(MEMOIRE_FILE):
        return pd.read_csv(MEMOIRE_FILE)
    return pd.DataFrame(columns=["date", "input", "concept", "coherence", "ghost_lvl"])

def sauver_memoire(u_input, concept, coherence, ghost_val):
    df = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), u_input, concept, round(coherence, 4), round(ghost_val, 2)]], 
                      columns=["date", "input", "concept", "coherence", "ghost_lvl"])
    df.to_csv(MEMOIRE_FILE, mode='a', header=not os.path.exists(MEMOIRE_FILE), index=False)

# --- 3. DICTIONNAIRE SÉMANTIQUE ---
MAPPING = {
    "RIEMANN": {"coord": np.array([0.8, 0.5, 0.4]), "ton": "🔬 ANALYTIQUE"},
    "ÉTHIQUE": {"coord": np.array([1.3, 0.7, 0.2]), "ton": "⚖️ EMPATHIQUE"},
    "ACTION":  {"coord": np.array([0.5, 0.2, 1.8]), "ton": "🔥 RADICAL"},
    "POÉSIE":  {"coord": np.array([0.3, 0.9, 0.8]), "ton": "🌙 ONYRIQUE"},
    "FANTÔME": {"coord": np.array([0.0, 0.0, 0.0]), "ton": "👻 INTUITION PROFONDE"}
}

# --- 4. MOTEUR TST GHOST PERMANENT ---
def ttu_engine_spectral(state, ghost_energy, K=2.0944, dt=0.01):
    m, c, d = state
    # Le fantôme crée une résonance stochastique qui empêche le blocage
    pression_fantome = ghost_energy * np.sin(m * d * 10) 
    
    dm = -d * np.sin(K * c)
    dc = 0.3 * (0.5 - c) + m * np.cos(K * d) + pression_fantome
    dd = 0.05 * (m * c) - 0.15 * d
    
    return state + np.array([dm, dc, dd]) * dt

# --- 5. INTERFACE ---
st.title("🧠 IA Souveraine : Analyse Spectrale & État Fantôme")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Paramètres Fantômes")
    ghost_perm = st.slider("Pression de Vide Permanente (Ghost)", 0.0, 2.0, 0.8)
    st.info("Un niveau élevé permet de détecter les vérités non-dites.")
    if st.button("Effacer la Mémoire"):
        if os.path.exists(MEMOIRE_FILE): os.remove(MEMOIRE_FILE)
        st.rerun()

if 'chat' not in st.session_state: st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.write(m["content"])

prompt = st.chat_input("Saisissez votre énoncé théorique ou émotionnel...")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.status("Analyse du spectre informationnel...", expanded=False):
            # Conditions initiales basées sur le texte
            phi = np.array([1.0 + (len(prompt)%20)/10, 0.2, 0.3])
            history = []
            
            # Simulation 3000 cycles pour une meilleure résolution spectrale
            for i in range(3000):
                phi = ttu_engine_spectral(phi, ghost_energy=ghost_perm)
                if i % 5 == 0: history.append(phi.copy())
            
            h_array = np.array(history)
            
            # Calcul de la proximité
            best_id = min(MAPPING.keys(), key=lambda k: np.linalg.norm(phi - MAPPING[k]["coord"]))
            sauver_memoire(prompt, best_id, phi[1], ghost_perm)

        st.write(f"### {MAPPING[best_id]['ton']}")
        
        # Réponse contextuelle simplifiée
        if best_id == "FANTÔME":
            st.write("L'état fantôme s'est cristallisé. Votre question touche à l'essence relationnelle du vide.")
        else:
            st.write(f"Stabilisation terminée sur l'attracteur **{best_id}**. Cohérence finale : `{phi[1]:.4f}`")

        # --- 6. ANALYSE SPECTRALE (FFT) ---
        st.subheader("📊 Spectre de Fréquence de la Pensée")
        
        # On fait la FFT sur la composante Cohérence (C)
        signal = h_array[:, 1]
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 0.01)[:N//2]
        amplitude = 2.0/N * np.abs(yf[0:N//2])

        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(x=xf, y=amplitude, name="Spectre C", line=dict(color='cyan')))
        fig_spec.update_layout(title="Résonance Harmonique (FFT)", xaxis_title="Fréquence", yaxis_title="Amplitude")
        st.plotly_chart(fig_spec)
        
        st.info("Un pic unique indique une pensée focalisée. Des pics multiples indiquent un paradoxe ou une intuition fantôme active.")

        # --- 7. EXPORT CSV ET VISUALISATION 3D ---
        col1, col2 = st.columns(2)
        
        with col1:
            df_export = pd.DataFrame(h_array, columns=['Mémoire', 'Cohérence', 'Dissipation'])
            csv = df_export.to_csv(index=True).encode('utf-8')
            st.download_button("📥 Télécharger le Chemin CSV", data=csv, file_name="path_spectral.csv", mime="text/csv")
        
        with col2:
            with st.expander("Voir la trajectoire 3D"):
                fig_3d = go.Figure(data=[go.Scatter3d(x=h_array[:,0], y=h_array[:,1], z=h_array[:,2], mode='lines', line=dict(color=h_array[:,1], colorscale='Viridis'))])
                st.plotly_chart(fig_3d)

    st.session_state.chat.append({"role": "assistant", "content": f"Stabilisé sur {best_id}"})
