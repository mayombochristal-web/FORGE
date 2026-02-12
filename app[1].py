import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONSTANTES ET MOTEUR TTU
# ==============================
PHI_SEUIL = 0.5088

def simulate_forge_trajectory(p_max):
    """Simule la montée vers la singularité Er-Au"""
    pressures = np.linspace(0, p_max, 100)
    phi_c = 0.65 + 0.35 * (1 - np.exp(-pressures / 80))
    phi_d = 1.0 * np.exp(-(phi_c - 0.5)**2 / 0.05) * (1 - phi_c)
    phi_m = phi_c ** 2 
    return pressures, phi_c, phi_d, phi_m

# ==============================
# NOUVEAU : MODULE HAMILTONIEN
# ==============================
def execute_hamiltonian_gate(phi_c, operation):
    """Simule une porte logique réversible en régime de singularité"""
    if phi_c < 0.95:
        return "❌ ERREUR : Cohérence insuffisante pour calcul réversible. Décohérence fatale."
    
    # Simulation d'une rotation de phase (Porte de Pauli ou Hadamard)
    if operation == "NOT (Pauli-X)":
        return "✅ OPÉRATION RÉUSSIE : Inversion de l'invariant informationnel (Phase 180°)."
    elif operation == "SUPERPOSITION (Hadamard)":
        return "✅ OPÉRATION RÉUSSIE : Holonomie répartie sur la variété MC³."
    return "Statut : Prêt."

# ==============================
# INTERFACE STREAMLIT
# ==============================
st.set_page_config(page_title="Forge TTU Singularité", layout="wide")
st.title("⚛️ ORDINATEUR DE SINGULARITÉ TTU-MC³")

# Sidebar
p_target = st.sidebar.slider("Pression de Forge (GPa)", 0.0, 500.0, 200.0)
gate_op = st.sidebar.selectbox("Opération Hamiltonienne (PEI)", ["NOT (Pauli-X)", "SUPERPOSITION (Hadamard)"])

# Calculs
pressures, phis_c, phis_d, phis_m = simulate_forge_trajectory(p_target)
current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]

# Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Cohérence (ΦC)", round(current_phi_c, 4))
col2.metric("Dissipation (ΦD)", f"{current_phi_d:.2e}")
col3.metric("Stase Temporelle", f"{1/(1-current_phi_c+1e-9):.1f}x")

# Exécution Hamiltonienne
st.subheader("🖥️ Processeur de Singularité")
result_gate = execute_hamiltonian_gate(current_phi_c, gate_op)

if current_phi_c >= 0.95:
    st.success(f"**Calcul Réversible Actif** : {result_gate}")
    st.info("L'ordinateur utilise l'holonomie de l'attracteur pour traiter l'information sans chaleur.")
else:
    st.error(f"**Échec du Calcul** : {result_gate}")

# Visualisation
st.subheader("📈 Dynamique de la Singularité")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pressures, phis_c, label="ΦC (Cohérence)", color="cyan", linewidth=2)
ax.fill_between(pressures, 0, phis_d * 10, color="red", alpha=0.2, label="ΦD (Dissipation x10)")
ax.axvline(x=200, color='yellow', linestyle='--', label="Seuil de Singularité")
ax.set_xlabel("Pression (GPa)")
ax.set_ylabel("Amplitude")
ax.legend()
st.pyplot(fig)
