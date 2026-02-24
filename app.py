import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import time

# ==========================================
# MOTEUR PHYSIQUE VTM (Backend Triadique)
# ==========================================

class TriadSystem:
    """
    Système dynamique triadique général (VTM v3).
    Équations :
        dM/dt = -α*M + β*C
        dC/dt = -γ*C + δ*M*D
        dD/dt =  η*C² - μ*D
    """
    def __init__(self, alpha=0.6, beta=1.2, gamma=0.7, delta=0.8, eta=0.5, mu=0.3):
        self.params = (alpha, beta, gamma, delta, eta, mu)

    def derivative(self, t, state):
        M, C, D = state
        a, b, g, d, e, m = self.params
        
        dM = -a * M + b * C
        dC = -g * C + d * M * D
        dD = e * C**2 - m * D
        return [dM, dC, dD]

# ==========================================
# INTERFACE UTILISATEUR VTM
# ==========================================

st.set_page_config(page_title="VTM v3 - Virtual Triadic Machine", layout="wide")

st.title("🧠 Virtual Triadic Machine (VTM v3)")
st.markdown("""
> **Calcul par Attracteur :** L'ordinateur devient un système physique simulé où le résultat 
> est la position finale dans l'espace des phases (Mémoire, Cohérence, Dissipation).
""")

# Barre latérale : Programmation de la Triade
with st.sidebar:
    st.header("⚙️ Programmation du Qtrit")
    alpha = st.slider("α (Dissipation M)", 0.1, 2.0, 0.6)
    beta = st.slider("β (Couplage M-C)", 0.1, 2.0, 1.2)
    gamma = st.slider("γ (Dissipation C)", 0.1, 2.0, 0.7)
    delta = st.slider("δ (Non-linéarité C-D)", 0.1, 2.0, 0.8)
    eta = st.slider("η (Génération D)", 0.1, 2.0, 0.5)
    mu = st.slider("μ (Évaporation D)", 0.1, 2.0, 0.3)
    
    st.divider()
    st.header("🚀 État Initial")
    m0 = st.number_input("ΦM Initial", value=1.0)
    c0 = st.number_input("ΦC Initial", value=0.5)
    d0 = st.number_input("ΦD Initial", value=0.1)
    
    t_max = st.number_input("Temps de calcul (T)", value=50)

# Exécution de la Simulation (Le "Calcul")
if st.button("⚡ Lancer la Convergence vers l'Attracteur"):
    system = TriadSystem(alpha, beta, gamma, delta, eta, mu)
    y0 = [m0, c0, d0]
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)

    # Résolution par intégration (Simule l'évolution du Qtrit)
    with st.spinner("Stabilisation de la Triade..."):
        sol = solve_ivp(system.derivative, t_span, y0, t_eval=t_eval, method='RK45')

    # Affichage des Résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📉 Évolution Temporelle")
        df = pd.DataFrame({
            'Temps': sol.t,
            'Mémoire (ΦM)': sol.y[0],
            'Cohérence (ΦC)': sol.y[1],
            'Dissipation (ΦD)': sol.y[2]
        })
        st.line_chart(df.set_index('Temps'))

    with col2:
        st.subheader("🌀 Espace des Phases (Attracteur)")
        fig = go.Figure(data=[go.Scatter3d(
            x=sol.y[0], y=sol.y[1], z=sol.y[2],
            mode='lines',
            line=dict(color=sol.t, colorscale='Viridis', width=4)
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='Mémoire (ΦM)',
                yaxis_title='Cohérence (ΦC)',
                zaxis_title='Dissipation (ΦD)'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Résultat final (Convergence)
    st.divider()
    m_final, c_final, d_final = sol.y[:, -1]
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Résultat ΦM (Attracteur)", round(m_final, 4))
    res_col2.metric("Stabilité ΦC", round(c_final, 4))
    res_col3.metric("Entropie Finale ΦD", round(d_final, 4))

    if abs(sol.y[0, -1] - sol.y[0, -2]) < 1e-4:
        st.success("✅ CALCUL TERMINÉ : Attracteur stable atteint.")
    else:
        st.warning("⚠️ SYSTÈME INSTABLE : Le calcul n'a pas encore convergé.")
