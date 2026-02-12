import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANTES PHYSIQUES TTU-MC³
# ==========================================================
HBAR = 6.5821e-16  # eV.s
PHI_SEUIL = 0.5088
PRESSION_CRITIQUE_REF = 200.0  # GPa référence

# ==========================================================
# MODÈLE DYNAMIQUE CONTINU (FORMULATION SCIENTIFIQUE)
# ==========================================================

def simulate_forge_trajectory(p_max, temperature):
    """
    Simulation continue différentiable
    basée sur un modèle d'approche exponentielle critique.
    """

    pressures = np.linspace(0, p_max, 300)

    # Paramètre thermique
    beta = 1 / (temperature + 1e-9)

    # Cohérence : croissance sigmoïde critique
    phi_c = 1 / (1 + np.exp(-(pressures - PRESSION_CRITIQUE_REF) / 40))

    # Dissipation : décroissance gaussienne modulée
    phi_d = np.exp(-beta * phi_c) * (1 - phi_c)

    # Mémoire : accumulation quadratique
    phi_m = phi_c**2

    # Gradient dΦC/dP
    dphi_dp = np.gradient(phi_c, pressures)

    # Indicateur de stabilité locale (approx jacobien scalaire)
    stability = 1 - np.abs(dphi_dp)

    return pressures, phi_c, phi_d, phi_m, dphi_dp, stability


# ==========================================================
# ANALYSE CRITIQUE
# ==========================================================

def extraction_pei(phi_c, phi_d, stability):
    """
    Verdict basé sur cohérence + dissipation + stabilité
    """

    if phi_c > 0.98 and phi_d < 1e-12 and stability > 0.9:
        return "SINGULARITÉ STABLE — Holonomie protégée"

    elif phi_c > PHI_SEUIL:
        return "RÉGIME CRITIQUE — Transition en cours"

    else:
        return "RÉGIME DISSIPATIF — Structure non stabilisée"


# ==========================================================
# INTERFACE
# ==========================================================

st.set_page_config(page_title="Forge TTU MC³ — Advanced", layout="wide")
st.title("⚛️ FORGE TTU — MOTEUR DYNAMIQUE AVANCÉ")

st.sidebar.header("Paramètres physiques")

p_target = st.sidebar.slider("Pression (GPa)", 0.0, 500.0, 200.0)
temperature = st.sidebar.number_input("Température (K)", value=0.05, format="%.4f")

# ==========================================================
# EXECUTION
# ==========================================================

pressures, phis_c, phis_d, phis_m, gradients, stability = simulate_forge_trajectory(
    p_target, temperature
)

current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]
current_phi_m = phis_m[-1]
current_grad = gradients[-1]
current_stability = stability[-1]

verdict = extraction_pei(current_phi_c, current_phi_d, current_stability)

# ==========================================================
# MÉTRIQUES
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("ΦC Cohérence", round(current_phi_c, 6))
col2.metric("ΦD Dissipation", f"{current_phi_d:.2e}")
col3.metric("Gradient Critique", f"{current_grad:.4e}")
col4.metric("Stabilité Locale", round(current_stability, 4))

# ==========================================================
# STATUT PHYSIQUE
# ==========================================================

st.subheader("Analyse du régime dynamique")

if "SINGULARITÉ STABLE" in verdict:
    st.success(verdict)
elif "CRITIQUE" in verdict:
    st.warning(verdict)
else:
    st.error(verdict)

# ==========================================================
# VISUALISATIONS SCIENTIFIQUES
# ==========================================================

st.subheader("Dynamique complète de la variété")

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

# ΦC & ΦD
ax[0].plot(pressures, phis_c)
ax[0].plot(pressures, phis_d)
ax[0].axhline(y=PHI_SEUIL, linestyle="--")
ax[0].set_xlabel("Pression (GPa)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Cohérence et Dissipation")

# Gradient critique
ax[1].plot(pressures, gradients)
ax[1].set_xlabel("Pression (GPa)")
ax[1].set_ylabel("dΦC/dP")
ax[1].set_title("Gradient critique")

# Stabilité locale
ax[2].plot(pressures, stability)
ax[2].set_xlabel("Pression (GPa)")
ax[2].set_ylabel("Indice stabilité")
ax[2].set_title("Stabilité dynamique locale")

plt.tight_layout()
st.pyplot(fig)

# ==========================================================
# EXPORT SCIENTIFIQUE
# ==========================================================

data = pd.DataFrame({
    "Pression": pressures,
    "Phi_C": phis_c,
    "Phi_D": phis_d,
    "Phi_M": phis_m,
    "Gradient": gradients,
    "Stabilite": stability
})

st.download_button(
    "Télécharger données scientifiques (CSV)",
    data.to_csv(index=False),
    "forge_ttu_donnees.csv"
)
