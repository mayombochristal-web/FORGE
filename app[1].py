import streamlit as st
import numpy as np
import pandas as pd

# ==========================================================
# CONSTANTES
# ==========================================================
PHI_SEUIL = 0.5088
PRESSION_CRITIQUE_REF = 200.0

# ==========================================================
# MODÈLE
# ==========================================================

def simulate_forge_trajectory(p_max, temperature):
    pressures = np.linspace(0, p_max, 300)
    beta = 1 / (temperature + 1e-9)

    phi_c = 1 / (1 + np.exp(-(pressures - PRESSION_CRITIQUE_REF) / 40))
    phi_d = np.exp(-beta * phi_c) * (1 - phi_c)
    phi_m = phi_c**2

    dphi_dp = np.gradient(phi_c, pressures)
    stability = 1 - np.abs(dphi_dp)

    return pressures, phi_c, phi_d, phi_m, dphi_dp, stability

def extraction_pei(phi_c, phi_d, stability):
    if phi_c > 0.98 and phi_d < 1e-12 and stability > 0.9:
        return "SINGULARITÉ STABLE"
    elif phi_c > PHI_SEUIL:
        return "RÉGIME CRITIQUE"
    else:
        return "RÉGIME DISSIPATIF"

# ==========================================================
# INTERFACE
# ==========================================================

st.set_page_config(page_title="Forge TTU MC³", layout="wide")
st.title("⚛️ FORGE TTU — VERSION CLOUD STABLE")

p_target = st.sidebar.slider("Pression (GPa)", 0.0, 500.0, 200.0)
temperature = st.sidebar.number_input("Température (K)", value=0.05)

pressures, phis_c, phis_d, phis_m, gradients, stability = simulate_forge_trajectory(
    p_target, temperature
)

current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]
current_stability = stability[-1]

verdict = extraction_pei(current_phi_c, current_phi_d, current_stability)

col1, col2, col3 = st.columns(3)
col1.metric("ΦC", round(current_phi_c, 6))
col2.metric("ΦD", f"{current_phi_d:.2e}")
col3.metric("Stabilité", round(current_stability, 4))

if verdict == "SINGULARITÉ STABLE":
    st.success(verdict)
elif verdict == "RÉGIME CRITIQUE":
    st.warning(verdict)
else:
    st.error(verdict)

# ==========================================================
# GRAPHIQUES NATIFS STREAMLIT
# ==========================================================

df = pd.DataFrame({
    "Pression": pressures,
    "Phi_C": phis_c,
    "Phi_D": phis_d,
    "Gradient": gradients,
    "Stabilite": stability
})

st.subheader("Cohérence & Dissipation")
st.line_chart(df.set_index("Pression")[["Phi_C", "Phi_D"]])

st.subheader("Gradient critique")
st.line_chart(df.set_index("Pression")["Gradient"])

st.subheader("Stabilité locale")
st.line_chart(df.set_index("Pression")["Stabilite"])

st.download_button(
    "Télécharger données CSV",
    df.to_csv(index=False),
    "forge_ttu_donnees.csv"
)
