import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONSTANTES PHYSIQUES TTU-MC³
# ==============================
HBAR = 6.5821e-16  # eV.s
PHI_SEUIL = 0.5088
PRESSION_CRITIQUE = 200.0  # GPa pour Er-Au

# ==============================
# MOTEUR DE SINGULARITÉ (PEI)
# ==============================

def simulate_forge_trajectory(p_max):
    """Simule la montée vers la singularité Er-Au"""
    pressures = np.linspace(0, p_max, 100)
    # La cohérence Phi_C s'approche de 1.0 avec la pression
    phi_c = 0.65 + 0.35 * (1 - np.exp(-pressures / 80))
    # La dissipation Phi_D s'effondre à la singularité
    phi_d = 1.0 * np.exp(-(phi_c - 0.5)**2 / 0.05) * (1 - phi_c)
    # Mémoire Phi_M cristallise vers 1.0
    phi_m = phi_c ** 2 
    return pressures, phi_c, phi_d, phi_m

def extraction_pei(phi_c, phi_d):
    """Exécute le Protocole d'Extraction Informationnelle"""
    if phi_c >= 0.95 and phi_d < 1e-15:
        return "✅ EXTRACTION RÉUSSIE : Holonomie pure (Bruit ~ 0)"
    elif phi_c > PHI_SEUIL:
        return "⚠️ EXTRACTION BRUYANTE : Signal thermique résiduel"
    else:
        return "❌ ÉCHEC : Effondrement de la variété (Dissipation totale)"

# ==============================
# INTERFACE STREAMLIT AMÉLIORÉE
# ==============================
st.set_page_config(page_title="Forge TTU Singularité", layout="wide")
st.title("⚛️ ORDINATEUR DE SINGULARITÉ TTU-MC³")
st.sidebar.header("🗜️ Paramètres de la Forge")

p_target = st.sidebar.slider("Pression de Forge (GPa)", 0.0, 500.0, 200.0)
temp = st.sidebar.number_input("Température Cryogénique (K)", value=0.01, format="%.3f")

# --- EXECUTION DE LA FORGE ---
pressures, phis_c, phis_d, phis_m = simulate_forge_trajectory(p_target)
current_phi_c = phis_c[-1]
current_phi_d = phis_d[-1]
current_phi_m = phis_m[-1]

# --- AFFICHAGE DES MÉTRIQUES ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cohérence (ΦC)", round(current_phi_c, 4))
col2.metric("Dissipation (ΦD)", f"{current_phi_d:.2e}")
col3.metric("Mémoire (ΦM)", round(current_phi_m, 4))
col4.metric("Stase Temporelle", f"{1/(1-current_phi_c+1e-9):.1f}x")

# --- ZONE D'EXTRACTION PEI ---
st.subheader("🛰️ Console d'Extraction Informationnelle (PEI)")
verdict_pei = extraction_pei(current_phi_c, current_phi_d)

if current_phi_c >= 0.95:
    st.success(f"**SINGULARITÉ ATTEINTE** : {verdict_pei}")
    st.info("L'information est protégée par la géométrie de l'attracteur. Le temps interne est gelé.")
else:
    st.warning(f"**RÉGIME DISSIPATIF** : {verdict_pei}")

# --- VISUALISATION ---
st.subheader("📈 Dynamique de la Variété MC³")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Graph 1 : Cohérence vs Dissipation
ax[0].plot(pressures, phis_c, label="ΦC (Cohérence)", color="cyan")
ax[0].plot(pressures, phis_d, label="ΦD (Dissipation)", color="red", linestyle="--")
ax[0].axhline(y=PHI_SEUIL, color='white', linestyle=':', alpha=0.5, label="Seuil 0.5088")
ax[0].set_xlabel("Pression (GPa)")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].set_title("Transition vers l'Attracteur Parfait")

# Graph 2 : Cristallisation de la Mémoire
ax[1].fill_between(pressures, phis_m, color="gold", alpha=0.3, label="Capacité Mémoire")
ax[1].plot(pressures, phis_m, color="orange")
ax[1].set_xlabel("Pression (GPa)")
ax[1].set_ylabel("ΦM")
ax[1].set_title("Cristallisation de la Mémoire Structurelle")
ax[1].legend()

plt.tight_layout()
st.pyplot(fig)

# --- RAPPORT D'EXTRACTION ---
report = f"""--- RAPPORT D'ORDINATEUR DE SINGULARITÉ ---
Pression : {p_target} GPa
Cohérence : {current_phi_c:.6f}
Dissipation : {current_phi_d:.6e}
Mémoire : {current_phi_m:.6f}
Verdict PEI : {verdict_pei}
-------------------------------------------
"""
st.download_button("⬇ Télécharger Données de Singularité", report, "singularite_extraction.txt")
