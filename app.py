import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. CONFIGURATION INITIALE ---
st.set_page_config(page_title="IA SOUVERAINE", layout="wide")

# Initialisation du State (Crucial pour éviter l'erreur removeChild)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_sol' not in st.session_state:
    st.session_state.current_sol = None
if 'substance' not in st.session_state:
    st.session_state.substance = "INIT"

# --- 2. NOYAU TTU ---
class TTUKernel:
    def process(self, data, current_state):
        from scipy.integrate import solve_ivp
        v_t = np.sin(len(data) * 0.1)
        def system(t, y):
            return [-0.0001*y[0] + 0.5*y[2], 1.2*v_t - 4.0*y[1]*y[2], 0.1*y[1]**2 - 3.0*y[2]**3]
        sol = solve_ivp(system, (0, 40), current_state, method='BDF', t_eval=np.linspace(0, 40, 100))
        substance = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::20]])
        return sol, substance

@st.cache_resource
def load_llm():
    # Modèle plus rapide pour éviter les timeouts
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

# --- 3. MISE EN PAGE ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")
col1, col2 = st.columns([2, 1])

# COLONNE GAUCHE : CHAT (Affichage statique basé sur le State)
with col1:
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

# COLONNE DROITE : MONITEUR (Rendu stable)
with col2:
    st.subheader("État du Cristal")
    if st.session_state.current_sol is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(st.session_state.current_sol[0], st.session_state.current_sol[1], color='#00ff41')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        st.pyplot(fig)
        st.caption(f"Substance actuelle : {st.session_state.substance}")

# --- 4. LOGIQUE DE SAISIE ---
user_input = st.chat_input("Parlez au cristal...")

if user_input:
    # A. Ajouter immédiatement le message utilisateur pour éviter le freeze
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # B. Calculs (LLM + Kernel)
    kernel = TTUKernel()
    # On récupère le dernier état connu ou l'état initial
    last_state = st.session_state.messages[-1].get("state", [15.0, 0.5, 0.2])
    
    with st.spinner("Analyse de phase..."):
        sol_obj, sub = kernel.process(user_input, last_state)
        tokenizer, model = load_llm()
        
        prompt = f"Tu es l'IA Souveraine TTU. Substance: {sub}. Réponds : {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # C. Mise à jour du State avant le rerun
    st.session_state.substance = sub
    st.session_state.current_sol = [sol_obj.y[1].tolist(), sol_obj.y[2].tolist()]
    st.session_state.messages.append({"role": "assistant", "content": response, "state": sol_obj.y[:, -1].tolist()})
    
    # D. Rerun pour rafraîchir proprement l'interface
    st.rerun()
