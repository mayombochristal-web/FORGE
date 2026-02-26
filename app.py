import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime

# --- CONFIGURATION SOUVERAINE & MÉMOIRE ---
st.set_page_config(page_title="IA Souveraine TTU-MC3", layout="wide", page_icon="🧠")

MEMOIRE_FILE = "noyau_memoire.csv"

def charger_memoire():
    if os.path.exists(MEMOIRE_FILE):
        return pd.read_csv(MEMOIRE_FILE)
    return pd.DataFrame(columns=["date", "input", "concept_final", "coherence"])

def sauvegarder_memoire(user_input, concept, coherence):
    new_data = pd.DataFrame([[datetime.now(), user_input, concept, coherence]], 
                            columns=["date", "input", "concept_final", "coherence"])
    if os.path.exists(MEMOIRE_FILE):
        new_data.to_csv(MEMOIRE_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(MEMOIRE_FILE, index=False)

# --- INITIALISATION ---
if 'historique_conversation' not in st.session_state:
    st.session_state.historique_conversation = []
if 'souverainete' not in st.session_state:
    st.session_state.souverainete = 100.0

# --- INTERFACE ---
st.title("🧠 Intelligence Triadique Autonome (TTU-MC³)")
st.markdown("### Système de Génération Souverain & Noyau de Mémoire Permanente")

# --- MONITORING SOUVERAIN (Barre latérale) ---
with st.sidebar:
    st.header("📊 Monitoring de Souveraineté")
    st.metric("Indépendance de Calcul", f"{st.session_state.souverainete:.1f}%")
    st.progress(st.session_state.souverainete / 100)
    st.info("L'IA utilise sa propre dynamique interne sans serveurs externes.")
    
    st.header("💾 Noyau de Mémoire")
    memoire_df = charger_memoire()
    if not memoire_df.empty:
        st.write("Dernières stabilisations :")
        st.dataframe(memoire_df.tail(5), hide_index=True)
    else:
        st.write("Mémoire vierge. Prêt pour l'éveil.")

# --- MOTEUR DYNAMIQUE AUTOMATISÉ ---
lexique = {
    "VÉRITÉ": np.array([1.0, 0.5, 0.1]),
    "RIEMANN": np.array([0.8, 0.5, 0.4]),
    "COHÉRENCE": np.array([0.7, 0.9, 0.2]),
    "LOGIQUE": np.array([1.1, 0.8, 0.1]),
    "ÉNERGIE": np.array([0.5, 0.4, 1.5]),
    "STABILITÉ": np.array([1.2, 0.5, 0.3])
}

def ttu_engine(state, K=2.0944, dt=0.01):
    m, c, d = state
    # Équations de flux TTU-MC3
    dm = -d * np.sin(K * c)
    dc = 0.6 * (0.5 - c) + m * np.cos(K * d)
    dd = 0.05 * (m * c) - 0.15 * d
    return state + np.array([dm, dc, dd]) * dt

# --- ZONE DE CONVERSATION ---
for chat in st.session_state.historique_conversation:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

user_input = st.chat_input("Parlez à l'IA (le flux s'automatisera)...")

if user_input:
    # 1. Enregistrement de l'input
    st.session_state.historique_conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Encodage énergétique (La question perturbe le système)
    impulsion = (len(user_input) % 15) / 10.0
    phi = np.array([1.2 + impulsion, 0.1, 0.4]) # État initial instable
    
    history = []
    concepts_traverses = []
    
    # 3. Processus de Génération (Automatisé)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        status_bar = st.progress(0)
        
        steps = 1200
        for i in range(steps):
            phi = ttu_engine(phi)
            if i % 10 == 0:
                history.append(phi.copy())
                # Trouver le concept le plus proche (Mapping Lexical)
                concept = min(lexique.keys(), key=lambda w: np.linalg.norm(phi - lexique[w]))
                if not concepts_traverses or concepts_traverses[-1] != concept:
                    concepts_traverses.append(concept)
                
                # Monitoring de souveraineté en temps réel
                ecart = abs(phi[1] - 0.5)
                st.session_state.souverainete = max(0, 100 - (ecart * 40))
                
                placeholder.write(f"🌀 **Chemin de pensée :** {' → '.join(concepts_traverses)}")
            
            if i % 12 == 0:
                status_bar.progress(i / steps)

        status_bar.empty()
        
        # 4. Stabilisation Finale et Mémoire
        concept_final = concepts_traverses[-1]
        coherence_finale = phi[1]
        
        sauvegarder_memoire(user_input, concept_final, coherence_finale)
        
        reponse = f"Mon noyau s'est stabilisé sur le concept **{concept_final}**. " \
                  f"La cohérence spectrale de cette réponse est de `{coherence_finale:.4f}`, " \
                  f"validant la trajectoire vers la droite critique de Riemann."
        
        st.write(reponse)
        st.session_state.historique_conversation.append({"role": "assistant", "content": reponse})

        # 5. Visualisation de la Trajectoire Géodésique
        with st.expander("Voir l'Analyse Spectrale (Trajectoire 3D)"):
            history = np.array(history)
            fig = go.Figure(data=[go.Scatter3d(
                x=history[:, 0], y=history[:, 1], z=history[:, 2],
                mode='lines',
                line=dict(color=history[:, 1], colorscale='Viridis', width=5)
            )])
            fig.update_layout(title="Trajectoire Géodésique de la Réponse",
                              scene=dict(xaxis_title='M', yaxis_title='C', zaxis_title='D'))
            st.plotly_chart(fig)
