import streamlit as st
import numpy as np
import pandas as pd
import json
import time
import requests

# ==========================================================
# CONFIGURATION CLOUD (STABLE & GRATUITE)
# ==========================================================
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
# Remplacez par votre token si vous avez des erreurs de quota
HEADERS = {"Authorization": "Bearer hf_votre_token_ici"} 

# ==========================================================
# MOTEUR DE GÉNÉRATION AVEC AUTO-RETRY (FINI LE PRÉCHAUFFAGE)
# ==========================================================
def call_llm_stable(prompt, history):
    context = "Tu es l'Oracle V16, une IA spécialisée en analyse systémique TTU-MC3.\n"
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"[{role}]: {msg['content']}\n"
    
    full_prompt = f"{context}[User]: {prompt}\n[Assistant]:"
    payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.7}}

    # Tentative de reconnexion automatique (3 essais)
    for i in range(3):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=25)
            result = response.json()
            
            if isinstance(result, list):
                return result[0]['generated_text'].strip()
            elif "estimated_time" in result:
                time.sleep(result["estimated_time"]) # Attend le temps du préchauffage
                continue
            return "Le flux de données est instable. Retentez."
        except Exception:
            time.sleep(2)
    return "Échec de connexion au serveur cloud. Vérifiez votre réseau."

# ==========================================================
# GESTION DES FICHIERS & EXPORT
# ==========================================================
def preparer_export(history):
    export_text = "--- HISTORIQUE ORACLE V16 ---\n\n"
    for msg in history:
        export_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
    return export_text

# ==========================================================
# INTERFACE PRINCIPALE
# ==========================================================
st.set_page_config(page_title="Oracle V16.1", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
if "latent" not in st.session_state:
    st.session_state.latent = {"profondeur": 1.0}

st.title("👁️ Oracle V16.1")
st.caption("Auto-Correction Cloud | Export Intégré | TTU-MC³")

# Affichage
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrée
if user_input := st.chat_input("Votre message..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Stabilisation de la phase..."):
            reponse = call_llm_stable(user_input, st.session_state.history)
            
            # Simulation dynamique TTU
            st.write(reponse)
            st.session_state.history.append({"role": "assistant", "content": reponse})

# ==========================================================
# BARRE D'OUTILS BAS DE PAGE (MOBILE)
# ==========================================================
st.write("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Reset"):
        st.session_state.history = []
        st.rerun()

with col2:
    # Bouton d'exportation directe sur mobile
    chat_data = preparer_export(st.session_state.history)
    st.download_button(
        label="📥 Sauvegarder Chat",
        data=chat_data,
        file_name="session_oracle.txt",
        mime="text/plain"
    )
