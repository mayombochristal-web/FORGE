import streamlit as st
import torch
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import time
import schedule
import threading
import numpy as np
from fpdf import FPDF # Assure-toi d'ajouter fpdf dans requirements.txt

# 1. IMPORTATION DU KERNEL (Doit être en haut)
from kernel import TTUKernel

# --- FONCTIONS UTILITAIRES (DÉFINIES AVANT L'USAGE) ---

def analyse_tension(sol):
    tension = np.var(sol.y[2]) * 100
    if tension > 5: return "CRITIQUE", "🔴"
    if tension > 2: return "AGITÉ", "🟡"
    return "STABLE", "🟢"

def export_report(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport de Phase TTU-MC3", ln=True, align='C')
    for item in history:
        # Nettoyage pour éviter les erreurs d'encodage PDF
        text = f"Q: {item['q']}\nSubstance: {item['s'][:30]}\n"
        pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="TTU-MC3 SOUVERAIN", layout="wide", page_icon="🌌")

@st.cache_resource
def load_brain():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

# Initialisation
if 'kernel' not in st.session_state:
    st.session_state.kernel = TTUKernel()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header("Commandes Autonomes")

# Le bouton de téléchargement utilise maintenant la fonction définie plus haut
if st.session_state.kernel.history:
    st.sidebar.download_button(
        label="📥 Télécharger Rapport PDF",
        data=export_report(st.session_state.kernel.history),
        file_name="rapport_phase.pdf",
        mime="application/pdf"
    )

if st.sidebar.button("🌙 Cycle de Sommeil Manuel"):
    # Logique de sommeil simplifiée
    st.sidebar.write("Apprentissage en cours...")

# --- CORPS DE L'APPLICATION ---
st.title("🌌 IA GÉNÉRATRICE TTU-MC3")

col1, col2 = st.columns([2, 1])

with col1:
    chat_container = st.container()
    user_input = st.chat_input("Posez votre question...")

with col2:
    status_placeholder = st.empty()
    plot_placeholder = st.empty()

if user_input:
    sol, meta = st.session_state.kernel.process(user_input)
    tension_label, emoji = analyse_tension(sol)
    
    # Mise à jour visuelle sécurisée
    status_placeholder.metric("État du Système", f"{emoji} {tension_label}")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sol.y[1], sol.y[2], color='#00ff41', lw=0.8)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')
    plot_placeholder.pyplot(fig)

    # Simulation de réponse (Remplace par ton code model.generate)
    response = "Analyse de phase terminée. La substance est scellée."
    
    with chat_container:
        st.chat_message("assistant").write(response)
