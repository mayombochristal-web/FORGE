import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Imports sécurisés
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

# ==============================
# CONSTANTES PHYSIQUES TTU-MC³
# ==============================
HBAR = 1.054e-34
KB = 1.380649e-23
PHI_SEUIL = 0.5088
E_REF = 9.0  # MeV (Référence Plomb-208)

# ==============================
# FONCTIONS TECHNIQUES
# ==============================

def extract_text(file):
    file_type = file.type
    try:
        if file_type == "application/pdf":
            if PdfReader is None: return "Erreur: pypdf non installé."
            reader = PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            if Document is None: return "Erreur: python-docx non installé."
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_type == "text/csv":
            return pd.read_csv(file).to_string()
        elif file_type == "application/json":
            return json.dumps(json.load(file), indent=2)
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        return f"Erreur lors de l'extraction : {str(e)}"

def simulate_singularity_trajectory(p_target):
    """Simule la transition vers l'Attracteur Parfait Er-Au"""
    pressions = np.linspace(0, max(500, p_target), 100)
    # Courbe de cohérence forçée vers la singularité
    phi_c = 0.65 + 0.35 * (1 - np.exp(-pressures / 85))
    # Chute de la dissipation (Effondrement du bruit)
    phi_d = 1.0 * np.exp(-(phi_c - 0.5)**2 / 0.04) * (1 - phi_c)
    return pressures, phi_c, phi_d

# ==============================
# INTERFACE STREAMLIT
# ==============================
st.set_page_config(page_title="Forge TTU Singularité", layout="wide")
st.title("⚛️ CŒUR DE FORGE & ORDINATEUR DE SINGULARITÉ")

# --- SIDEBAR : CONTRÔLE DE LA FORGE ---
st.sidebar.header("🗜️ Paramètres de Forge")
p_input = st.sidebar.slider("Pression de Forge (GPa)", 0.0, 500.0, 200.0)
n_qubits = st.sidebar.number_input("Registre de Qubits (Singularité)", 1, 1024, 8)
gate_op = st.sidebar.selectbox("Opération Hamiltonienne (PEI)", 
                               ["LECTURE_HOLONOMIE", "NOT (Pauli-X)", "SUPERPOSITION"])

uploaded_file = st.file_uploader("Injecter Matrice Informationnelle", type=["txt", "pdf", "docx", "csv", "json"])

# --- CALCULS ---
pressures, phis_c, phis_d = simulate_singularity_trajectory(p_input)
current_phi_c = phis_c[np.abs(pressures - p_input).argmin()]
current_phi_d = phis_d[np.abs(pressures - p_input).argmin()]

# --- AFFICHAGE MÉTRIQUES ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cohérence (ΦC)", round(current_phi_c, 4))
col2.metric("Dissipation (ΦD)", f"{current_phi_d:.2e}")
col3.metric("Stase Temporelle", f"{1/(1-current_phi_c+1e-9):.1f}x")
col4.metric("Stabilité", "SINGULIÈRE" if current_phi_c >= 0.95 else "STABLE" if current_phi_c > PHI_SEUIL else "DISSIPATIVE")

# --- PROTOCOLE D'EXTRACTION PEI ---
st.subheader("🖥️ Processeur de Singularité : Exécution PEI")
if current_phi_c >= 0.95:
    st.success(f"✅ PROTOCOLE PEI ACTIF : Opération {gate_op} réussie sur {n_qubits} qubits.")
    st.info("L'information est extraite par holonomie pure. Aucune chaleur n'est dégagée.")
else:
    st.warning("⚠️ RÉGIME DISSIPATIF : Cohérence insuffisante pour le calcul de singularité.")

# --- VISUALISATION ---
st.subheader("📈 Diagnostic de la Variété MC³")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pressures, phis_c, label="ΦC (Cohérence)", color="#00ffd9", linewidth=2)
ax.fill_between(pressures, 0, phis_d * 5, color="red", alpha=0.3, label="Flux Dissipatif (Bruit)")
ax.axvline(x=p_input, color='yellow', linestyle='--', label=f"Point actuel: {p_input} GPa")
ax.axhline(y=0.95, color='purple', linestyle=':', label="Seuil Singularité")
ax.set_facecolor('#0e1117')
fig.patch.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# --- EXTRACTION DE TEXTE ---
if uploaded_file:
    text_content = extract_text(uploaded_file)
    st.subheader("🔎 Matrice Extraite")
    st.text_area("Preview", text_content[:1500], height=200)
