import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TTU-MC3 Cyber-Forge", layout="wide")

# --- STYLE CSS POUR LE BRANDING ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .payment-box { 
        padding: 20px; 
        border: 2px solid #ff4b4b; 
        border-radius: 10px; 
        background-color: #1e1e1e;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TITRE ET CONTACT ---
st.title("🛡️ TTU-MC³ : Unified Cyber-Forge")
st.sidebar.markdown(f"""
### 📞 Contact & Support
**Email :** [mayombochristal@gmail.com](mailto:mayombochristal@gmail.com)
""")

# --- MODULES DE PAIEMENT ---
def display_payment_info():
    st.markdown("""
    <div class="payment-box">
        <h3>💳 ACTIVER LA FORGE (SERVICES PREMIUM)</h3>
        <p>Pour débloquer le cryptage haute cohérence et les audits complets :</p>
        <p><b>Gabon 🇬🇦 (Airtel/Moov) : +241 77 76 54 96</b></p>
        <p><b>Congo 🇨🇬 (Airtel/Moov) : +241 65 43 00 33</b></p>
        <p><i>Envoyez la preuve de transfert à l'email ci-dessus pour recevoir votre clé d'activation.</i></p>
    </div>
    """, unsafe_allow_html=True)

# --- LOGIQUE TTU-MC3 ---
def diode_chua(x):
    m0, m1 = -1.143, -0.714
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

def dynamics(state, t, a, b):
    x, y, z = state
    return [a*(y - x - diode_chua(x)), x - y + z, -b*y]

# --- INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🔎 Audit Gratuit", "🔐 Cryptage (Premium)", "🧪 Forge Matérielle"])

with tab1:
    st.header("Analyseur de Résilience (Pentesting)")
    st.write("Évaluez si votre système est un **Rocher** ou une **Fleur de Givre**.")
    val_liaison = st.number_input("Énergie de liaison mesurée (MeV)", 0.0, 10.0, 4.5)
    
    phi_c = val_liaison / 9.0
    if phi_c < 0.5088:
        st.error(f"⚠️ VULNÉRABILITÉ DÉTECTÉE : Cohérence {phi_c:.4f} < 0.5088")
        st.write("Votre structure informationnelle est instable face aux attaques par chaos.")
    else:
        st.success(f"✅ SYSTÈME ROBUSTE : Cohérence {phi_c:.4f} > 0.5088")

with tab2:
    st.header("Tunnel de Communication Chaotique")
    display_payment_info()
    st.warning("Le module de cryptage par synchronisation de chaos est verrouillé.")
    if st.text_input("Entrez votre clé d'activation payante") == "TTU-2026-PRO":
        st.success("Accès autorisé à la Forge Virtuelle.")
        # Le code de cryptage s'exécute ici
        

with tab3:
    st.header("Forge Acoustique : Chaux de Carbure")
    st.info("Utilisez cette section pour stabiliser vos matériaux (Résonance 19.605 Hz).")
    st.write("Contactez **mayombochristal@gmail.com** pour les protocoles industriels complets.")
    

st.divider()
st.write("© 2026 Start-up TTU-MC³. Tous droits réservés.")
