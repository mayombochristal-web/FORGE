import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION ET MOTEUR TTU
# ==============================
PHI_SEUIL = 0.5088

def simulate_forge_trajectory(p_max):
    """Simule la montée vers la singularité Er-Au"""
    pressures = np.linspace(0, p_max, 100)
    phi_c = 0.65 + 0.35 * (1 - np.exp(-pressures / 80))
    phi_d = 1.0 * np.exp(-(phi_c - 0.5)**2 / 0.05) * (1 - phi_c)
    phi_m = phi_c ** 2 
    return pressures, phi_c, phi_d, phi_m

def execute_multi_qubit_logic(phi_c, n_qubits):
    """Simule un registre de qubits protégés par l'attracteur"""
    if phi_c < 0.95:
        return "❌ ERREUR : Cohérence insuffisante. Effondrement du registre."
    
    capacité = n_qubits * phi_c
    return f"✅ REGISTRE ACTIF : {n_qubits} Qubits d'attracteurs stabilisés. Capacité : {capacité:.2f} Shannons."

# ==============================
# INTERFACE STREAMLIT
# ==============================
st.set_page_config(page_title="Forge TTU Multi-Qubit", layout="wide")
st.title("⚛️ PROCESSEUR MULTI-QUBITS DE SINGULARITÉ")

# Sidebar
st.sidebar.header("🗜️ Contrôle de la Forge")
p_target = st.sidebar.slider("Pression de Forge (GPa)", 0.0, 500.0, 200.0)
n_qubits = st.sidebar.number_input("Nombre de Qubits d'Attracteurs", min_value=1, max_value=1024, value=8)

# Calculs
pressures, phis_c, phis_d, phis_m = simulate_forge_trajectory(p_target)
current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]

# Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Cohérence (ΦC)", round(current_phi_c, 4))
col2.metric("Dissipation (ΦD)", f"{current_phi_d:.2e}")
col3.metric("Stase Temporelle", f"{1/(1-current_phi_c+1e-9):.1f}x")

# --- CONSOLE DE CALCUL ---
st.subheader("🖥️ État du Registre de Singularité (PEI)")
result_logic = execute_multi_qubit_logic(current_phi_c, n_qubits)

if current_phi_c >= 0.95:
    st.success(f"**Ordinateur de Singularité Opérationnel** : {result_logic}")
    st.info("Chaque qubit est une trajectoire stable dans l'attracteur Er-Au.")
else:
    st.error(f"**Alerte Décohérence** : {result_logic}")

# --- VISUALISATION ---
st.subheader("📈 Diagnostic de la Variété Informationnelle")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pressures, phis_c, label="ΦC (Cohérence)", color="#00ffd9", linewidth=2.5)
ax.fill_between(pressures, 0, phis_d * 5, color="#ff4b4b", alpha=0.3, label="Flux Dissipatif")
ax.axvline(x=200, color='yellow', linestyle='--', label="Seuil Singularité")
ax.set_facecolor('#0e1117')
fig.patch.set_facecolor('#0e1117')
ax.set_xlabel("Pression (GPa)", color="white")
ax.set_ylabel("Amplitude", color="white")
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)
