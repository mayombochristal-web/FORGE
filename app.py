import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# --- CONFIGURATION TTU-MC3 ---
st.set_page_config(page_title="IA Génératrice TTU-MC3", layout="wide")

st.title("🧠 IA Génératrice Autonome (Cadre TTU-MC³ / TST)")
st.markdown("""
*Ce prototype remplace le calcul probabiliste (GPT) par un **calcul par attracteur**. 
L'IA converge physiquement vers la solution la plus stable sur la droite critique de Riemann.*
""")

# --- PARAMÈTRES DYNAMIQUES DANS LA BARRE LATÉRALE ---
st.sidebar.header("Paramètres de la Triade")
dt = st.sidebar.slider("Pas de temps (dt)", 0.001, 0.05, 0.01)
K = st.sidebar.slider("Invariant Z3 (K)", 0.0, 4.0, 2.0944) # 2pi/3
steps = st.sidebar.number_input("Nombre de cycles", 500, 5000, 1500)

# Lexique sémantique (Ancres de l'attracteur)
lexique = {
    "CHAOS":      np.array([1.5, 0.1, 0.8]),
    "STRUCTURE":  np.array([1.2, 0.3, 0.2]),
    "LOGIQUE":    np.array([1.0, 0.8, 0.1]),
    "RIEMANN":    np.array([0.8, 0.5, 0.4]), # Cible Droite Critique 0.5
    "ÉQUILIBRE":  np.array([0.7, 0.7, 0.7]),
    "ÉNERGIE":    np.array([0.4, 0.5, 1.4])
}

# --- MOTEUR DYNAMIQUE ---
def ttu_flow(state, K):
    m, c, d = state
    dm = -d * np.sin(K * c)
    dc = 0.6 * (0.5 - c) + m * np.cos(K * d)
    dd = 0.05 * (m * c) - 0.15 * d
    return np.array([dm, dc, dd])

def get_closest_concept(state):
    word = min(lexique.keys(), key=lambda w: np.linalg.norm(state - lexique[w]))
    return word

# --- EXÉCUTION DE LA GÉNÉRATION ---
if st.button("Lancer le Chemin de Pensée"):
    phi = np.array([1.4, 0.1, 0.3]) # État initial (Bruit)
    history = []
    thoughts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(steps):
        phi += ttu_flow(phi, K) * dt
        
        if i % 10 == 0:
            word = get_closest_concept(phi)
            history.append(phi.copy())
            if not thoughts or thoughts[-1] != word:
                thoughts.append(word)
            status_text.text(f"Pensée actuelle : {word} | Cohérence : {phi[1]:.4f}")
        
        if i % (steps//100) == 0:
            progress_bar.progress(i / steps)

    history = np.array(history)

    # --- AFFICHAGE DES RÉSULTATS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Migration Lexicale")
        st.success(" -> ".join(thoughts))
        
        # Graphique des composantes
        df = pd.DataFrame(history, columns=['Mémoire', 'Cohérence', 'Dissipation'])
        st.line_chart(df)
        st.info("Observez la Cohérence (orange) se stabiliser vers 0.5 (Riemann).")

    with col2:
        st.subheader("Portrait de Phase 3D")
        fig = go.Figure(data=[go.Scatter3d(
            x=history[:, 0], y=history[:, 1], z=history[:, 2],
            mode='lines',
            line=dict(color=history[:, 1], colorscale='Viridis', width=5)
        )])
        fig.update_layout(scene=dict(xaxis_title='M', yaxis_title='C', zaxis_title='D'))
        st.plotly_chart(fig)

    st.balloons()
