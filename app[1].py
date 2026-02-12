
import streamlit as st
import pandas as pd
import json
import math
from PyPDF2 import PdfReader
from docx import Document

# ============================
# CONSTANTES PHYSIQUES
# ============================

HBAR = 1.054e-34
KB = 1.380649e-23
PHI_SEUIL = 0.5088
E_REF = 9.0  # MeV référence plomb-208

# ============================
# EXTRACTION MULTI-FORMAT
# ============================

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string()

    elif file.type == "application/json":
        data = json.load(file)
        return json.dumps(data, indent=2)

    else:
        return file.read().decode("utf-8")


# ============================
# MOTEUR TTU-MC³
# ============================

def compute_phi_coherence(energy_liaison):
    return energy_liaison / E_REF

def compute_dissipation(phi_c, tau=1e-12):
    return (HBAR / tau) * (phi_c / PHI_SEUIL) ** 2

def compute_internal_time(phi_c, temperature=300):
    if phi_c == 0:
        return float("inf")
    return (KB * temperature) / phi_c


# ============================
# INTERFACE STREAMLIT
# ============================

st.set_page_config(layout="wide")
st.title("⚛️ CŒUR DE FORGE TTU — Version Scientifique Locale")

uploaded_file = st.file_uploader(
    "Injecter Matrice",
    type=["txt", "pdf", "docx", "csv", "json"]
)

if uploaded_file:

    text_content = extract_text(uploaded_file)

    st.subheader("🔎 Contenu extrait")
    st.text_area("Preview", text_content[:2000], height=250)

    st.subheader("⚙️ Paramètres Physiques")

    energy = st.number_input("Énergie de liaison (MeV)", value=7.03)
    temperature = st.number_input("Température (K)", value=300)

    if st.button("⚡ Lancer la Forge TTU"):

        phi_c = compute_phi_coherence(energy)
        phi_d = compute_dissipation(phi_c)
        t_internal = compute_internal_time(phi_c, temperature)

        st.subheader("📊 Résultats TTU")

        col1, col2, col3 = st.columns(3)
        col1.metric("ΦC (Cohérence)", round(phi_c, 4))
        col2.metric("ΦD (Dissipation)", f"{phi_d:.2e}")
        col3.metric("Temps interne", f"{t_internal:.2e}")

        if phi_c > PHI_SEUIL:
            st.success("✅ SYSTÈME PHYSIQUE STABLE (ΦC > 0.5088)")
        else:
            st.error("⚠️ SYSTÈME THERMIQUE / BRUIT")

        report = f"""
--- RAPPORT TTU-MC³ ---

Énergie liaison: {energy} MeV
ΦC = {phi_c}
ΦD = {phi_d}
Temps interne = {t_internal}

Seuil critique = {PHI_SEUIL}

Conclusion :
{"Stable" if phi_c > PHI_SEUIL else "Instable / Dissipatif"}

-------------------------
"""

        st.download_button(
            "⬇ Télécharger Rapport",
            report,
            file_name="rapport_ttu.txt"
        )
