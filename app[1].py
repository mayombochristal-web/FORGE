import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONSTANTES ET MOTEUR TTU-MC³
# ==============================
PHI_SEUIL = 0.5088

def simulate_forge_trajectory(p_max):
    """Simule la montée vers la singularité Er-Au"""
    pressures = np.linspace(0, p_max, 100)
    # La cohérence Phi_C tend vers 1.0 (Singularité)
    phi_c = 0.65 + 0.35 * (1 - np.exp(-pressures / 80))
    # La dissipation s'effondre, gelant la flèche du temps
    phi_d = 1.0 * np.exp(-(phi_c - 0.5)**2 / 0.05) * (1 - phi_c)
    phi_m = phi_c ** 2 
    return pressures, phi_c, phi_d, phi_m

# ==============================
# MODULE DE CALCUL HAMILTONIEN
# ==============================
def execute_hamiltonian_gate(phi_c, operation):
    """Exécute une opération logique réversible via le protocole PEI"""
    if phi_c < 0.95:
        return "❌ ÉCHEC : Décohérence fatale (ΦC < 0.95). L'information s'évapore."
    
    # Simulation de portes quantiques topologiques
    if operation == "NOT (Pauli-X)":
        return "✅ RÉUSSI : Inversion d'état par holonomie géométrique."
    elif operation == "SUPERPOSITION (Hadamard)":
        return "✅ RÉUSSI : État réparti sur la variété informationnelle."
    return "En attente d'instruction..."

# ==============================
# INTERFACE UTILISATEUR
# ==============================
st.set_page_config(page_title="Forge TTU Singularité", layout="wide")
st.title("⚛️ ORDINATEUR DE SINGULARITÉ TTU-MC³")

# Sidebar de contrôle
st.sidebar.header("🗜️ Paramètres de la Forge")
p_target = st.sidebar.slider("Pression de Forge (GPa)", 0.0, 500.0, 200.0)
gate_op = st.sidebar.selectbox("Opération Hamiltonienne (PEI)", 
                               ["NOT (Pauli-X)", "SUPERPOSITION (Hadamard)"])

# Calcul des données
pressures, phis_c, phis_d, phis_m = simulate_forge_trajectory(p_target)
current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]

# Affichage des métriques de singularité
col1, col2, col3 = st.columns(3)
col1.metric("Cohérence (ΦC)", round(current_phi_c, 4))
col2.metric("Dissipation (ΦD)", f"{current_phi_d:.2e}")
col3.metric("Stase Temporelle", f"{1/(1-current_phi_c+1e-9):.1f}x")

# --- CONSOLE DE CALCUL ---
st.subheader("🖥️ Processeur de Singularité (PEI)")
result_gate = execute_hamiltonian_gate(current_phi_c, gate_op)

if current_phi_c >= 0.95:
    st.success(f"**Calcul Hamiltonien Actif** : {result_gate}")
    st.info("Le système opère dans l'Attracteur Parfait : aucune chaleur n'est générée.")
else:
    st.error(f"**Rupture de Cohérence** : {result_gate}")

# --- VISUALISATION ---
st.subheader("📈 Diagnostic de la Variété MC³")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(pressures, phis_c, label="ΦC (Cohérence)", color="cyan", linewidth=2.5)
ax.fill_between(pressures, 0, phis_d * 5, color="red", alpha=0.3, label="Flux Dissipatif (Bruit)")
ax.axvline(x=200, color='yellow', linestyle='--', label="Seuil de Singularité")
ax.set_xlabel("Pression (GPa)")
ax.set_ylabel("Amplitude")
ax.set_facecolor('#0e1117')
fig.patch.set_facecolor('#0e1117')
ax.legend()
st.pyplot(fig)
