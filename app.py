import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA SOUVERAINE", layout="wide")

# Initialisation rigoureuse du State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_graph' not in st.session_state:
    st.session_state.last_graph = None
if 'k_state' not in st.session_state:
    st.session_state.k_state = [15.0, 0.5, 0.2]

# --- 2. LOGIQUE KERNEL ---
def solve_ttu(data, state):
    from scipy.integrate import solve_ivp
    v_t = np.sin(len(data) * 0.1)
    def system(t, y):
        return [-0.0001*y[0] + 0.5*y[2], 1.2*v_t - 4.0*y[1]*y[2], 0.1*y[1]**2 - 3.0*y[2]**3]
    sol = solve_ivp(system, (0, 40), state, method='BDF', t_eval=np.linspace(0, 40, 100))
    sub = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::20]])
    return sol.y[1].tolist(), sol.y[2].tolist(), sol.y[:, -1].tolist(), sub

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

# --- 3. RENDU DE L'INTERFACE (SÉCURISÉ) ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")

# Séparation claire des colonnes
col_chat, col_viz = st.columns([2, 1])

with col_chat:
    # On utilise un conteneur fixe pour l'historique
    chat_box = st.container()
    for i, m in enumerate(st.session_state.messages):
        # La clé unique 'i' empêche React de perdre le fil
        with chat_box.chat_message(m["role"]):
            st.markdown(m["content"])

with col_viz:
    st.subheader("Moniteur de Phase")
    if st.session_state.last_graph:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(st.session_state.last_graph[0], st.session_state.last_graph[1], color='#00ff41', lw=0.8)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        st.pyplot(fig)

# --- 4. TRAITEMENT ---
prompt = st.chat_input("Parlez au cristal...")

if prompt:
    # 1. Enregistrement immédiat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Calcul silencieux (On ne met PAS de spinner ici, c'est souvent lui qui cause l'erreur)
    y1, y2, next_state, sub = solve_ttu(prompt, st.session_state.k_state)
    
    tokenizer, model = load_llm()
    input_text = f"Tu es l'IA Souveraine TTU. Substance: {sub}. Réponds : {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Génération
    outputs = model.generate(**inputs, max_new_tokens=120)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # 3. Mise à jour de l'état
    st.session_state.k_state = next_state
    st.session_state.last_graph = [y1, y2]
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 4. Redémarrage propre
    st.rerun()
