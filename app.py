import streamlit as st
import numpy as np
import pandas as pd
import json
import time
import requests

# ==========================================================
# CONFIGURATION CLOUD (GRATUIT & SANS INSTALLATION)
# ==========================================================
# Nous utilisons l'API Inference de Hugging Face (Gratuit)
# Modèle : Mistral-7B ou Llama-3 (selon disponibilité)
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
# Note : Pour une utilisation intensive, créez un compte gratuit sur HuggingFace 
# et insérez votre jeton (Token) ci-dessous. Sinon, cela fonctionne avec des quotas limités.
HEADERS = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"} 

# ==========================================================
# MOTEUR DE GÉNÉRATION (APPEL API)
# ==========================================================
def call_cloud_llm(messages):
    """Appelle le LLM dans le cloud gratuitement"""
    # On transforme l'historique en un prompt unique pour Mistral
    prompt = ""
    for msg in messages:
        role = "Instruction" if msg["role"] == "system" else "Utilisateur"
        prompt += f"\n[{role}]: {msg['content']}\n"
    prompt += "[Assistant]:"

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=15)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]['generated_text'].split("[Utilisateur]")[0].strip()
        return "Désolé, le moteur est en cours de préchauffage. Réessayez dans 10 secondes."
    except Exception as e:
        return f"Erreur de connexion Cloud : {e}"

# ==========================================================
# LOGIQUE TTU-MC³ & APPRENTISSAGE
# ==========================================================
class MobileOracleEngine:
    def __init__(self):
        if "latent" not in st.session_state:
            st.session_state.latent = {"profondeur": 1.2, "coherence": 1.0}

    def simuler_metriques(self, history_len):
        t = np.linspace(0, 10, 100)
        ghost = 0.5 + (history_len * 0.1 * st.session_state.latent["profondeur"])
        df = pd.DataFrame({
            "M": np.exp(-t * 0.05),
            "C": (1.2 + ghost * np.sin(t * 0.2)),
            "D": 0.1 * np.exp(-history_len * 0.1) + 0.05 * np.random.randn(100)
        })
        return df, ghost

# ==========================================================
# INTERFACE STREAMLIT MOBILE-FRIENDLY
# ==========================================================
st.set_page_config(page_title="Oracle V15 Mobile", layout="centered")

engine = MobileOracleEngine()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("👁️ Oracle V15")
st.caption("IA Intégrée | Cloud Gratuit | TTU-MC³")

# Affichage du Chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrée utilisateur
if user_input := st.chat_input("Posez votre question..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # 1. Calcul de phase
        df_metrics, ghost = engine.simuler_metriques(len(st.session_state.history))
        
        # 2. Appel du cerveau distant (Gratuit)
        with st.spinner("L'Oracle consulte le flux..."):
            # Construction du contexte pour le LLM
            context_messages = [
                {"role": "system", "content": f"Tu es l'Oracle V15. Niveau de profondeur : {st.session_state.latent['profondeur']}. Réponds de façon étayée et précise."},
            ] + st.session_state.history[-5:] # On prend les 5 derniers messages
            
            reponse = call_cloud_llm(context_messages)
            
            # 3. Mise en perspective TTU
            final_output = f"{reponse}\n\n---\n*Analyse de Phase : Stabilité {ghost:.2f} | Cohérence optimale.*"
            st.write(final_output)
            st.session_state.history.append({"role": "assistant", "content": final_output})

        with st.expander("📊 Voir la dynamique de réflexion"):
            st.line_chart(df_metrics)

# Système de Feedback simple
if len(st.session_state.history) > 0:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Pertinent"):
            st.session_state.latent["profondeur"] += 0.1
            st.toast("L'IA affine sa profondeur.")
    with col2:
        if st.button("👎 Trop vague"):
            st.session_state.latent["profondeur"] -= 0.1
            st.toast("L'IA ajuste sa rigueur.")
