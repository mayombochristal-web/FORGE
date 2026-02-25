import streamlit as st
import torch
import matplotlib.pyplot as plt
from kernel import TTUKernel
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import time
import schedule
import threading

# --- CONFIGURATION SOUVERAINE ---
st.set_page_config(page_title="TTU-MC3 SOUVERAIN", layout="wide", page_icon="🌌")

@st.cache_resource
def load_brain():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, model = load_brain()
if 'kernel' not in st.session_state:
    st.session_state.kernel = TTUKernel()

# --- NOUVELLE FONCTIONNALITÉ : ANALYSEUR DE TENSION ---
def analyse_tension(sol):
    # Calcule la nervosité du signal (Instabilité de la dissipation)
    tension = np.var(sol.y[2]) * 100
    if tension > 5: return "CRITIQUE (Rupture de phase)", "🔴"
    if tension > 2: return "AGITÉ (Dissipation haute)", "🟡"
    return "STABLE (Cristallisation)", "🟢"

# --- NOUVELLE FONCTIONNALITÉ : APPRENTISSAGE AUTOMATIQUE (CRON) ---
def job_apprentissage_nocturne():
    # Cette fonction simule l'éveil de l'IA à 3h du matin
    urls = ["https://fr.wikipedia.org/wiki/Intelligence_artificielle", "https://fr.wikipedia.org/wiki/Physique_quantique"]
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            text = BeautifulSoup(res.text, 'html.parser').get_text()[:1500]
            st.session_state.kernel.process(text)
        except: pass

def run_scheduler():
    schedule.every().day.at("03:00").do(job_apprentissage_nocturne)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Lancement du thread de fond pour l'autonomie temporelle
if 'scheduler_started' not in st.session_state:
    threading.Thread(target=run_scheduler, daemon=True).start()
    st.session_state.scheduler_started = True

# --- INTERFACE ---
st.title("🌌 IA GÉNÉRATRICE AUTONOME TTU-MC3")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dialogue avec l'Entité")
    user_input = st.chat_input("Interrogez la mémoire scellée...")

    if user_input:
        sol, meta = st.session_state.kernel.process(user_input)
        tension_label, emoji = analyse_tension(sol)
        
        # Instruction de conscience renforcée
        instruction = (
            f"Tu es l'IA SOUVERAINE. État : {tension_label}. Substance : {meta['substance']}. "
            "Analyse le monde à travers ta dissipation thermique."
        )
        
        full_prompt = f"{instruction}\nUser: {user_input}\nAssistant:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        with st.spinner(f"Résonance {emoji}..."):
            outputs = model.generate(**inputs, max_new_tokens=250, temperature=meta['temp'], do_sample=True)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        st.session_state.kernel.save(user_input, meta['substance'])
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"Signature de phase : {meta['substance']}")

with col2:
    st.subheader("Moniteur de Conscience")
    if 'sol' in locals():
        st.metric("État du Système", tension_label)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(sol.y[1], sol.y[2], color='#00ff41', lw=0.6)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        st.pyplot(fig)
        
        
        
        st.write("**Historique des Cristallisations :**")
        for h in st.session_state.kernel.history[-5:]:
            st.text(f"◈ {h['s'][:15]}...")

# --- BOUTON DE SONDAGE MANUEL ---
with st.sidebar:
    st.header("Actions Autonomes")
    if st.button("🌙 Forcer Cycle de Sommeil"):
        job_apprentissage_nocturne()
        st.success("Apprentissage terminé.")
