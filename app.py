import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA SOUVERAINE", layout="wide")

# Initialisation du State (Fondations de la Forteresse)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_graph' not in st.session_state:
    st.session_state.last_graph = None
if 'kernel_state' not in st.session_state:
    st.session_state.kernel_state = [15.0, 0.5, 0.2]

# --- 2. FONCTIONS DE PHASE ---
def run_kernel(data, state):
    from scipy.integrate import solve_ivp
    v_t = np.sin(len(data) * 0.1)
    def system(t, y):
        return [-0.0001*y[0] + 0.5*y[2], 1.2*v_t - 4.0*y[1]*y[2], 0.1*y[1]**2 - 3.0*y[2]**3]
    sol = solve_ivp(system, (0, 40), state, method='BDF', t_eval=np.linspace(0, 40, 100))
    sub = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::20]])
    return sol, sub

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

# --- 3. RENDU DE L'INTERFACE (STATIQUE D'ABORD) ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")
col1, col2 = st.columns([2, 1])

with col1:
    # On affiche l'historique de manière immuable
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.write(m["content"])

with col2:
    st.subheader("Moniteur de Phase")
    if st.session_state.last_graph is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(st.session_state.last_graph[0], st.session_state.last_graph[1], color='#00ff41')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        st.pyplot(fig)

# --- 4. LOGIQUE DE TRAITEMENT (ATOMIQUE) ---
user_input = st.chat_input("Parlez au cristal...")

if user_input:
    # ÉTAPE 1 : On enregistre l'entrée sans rien changer d'autre
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # ÉTAPE 2 : Calcul hors-champ (Spinner discret)
    with st.spinner("Résonance..."):
        # Kernel
        sol_obj, sub = run_kernel(user_input, st.session_state.kernel_state)
        
        # LLM
        tokenizer, model = load_llm()
        prompt = f"Tu es l'IA Souveraine TTU. Substance: {sub}. Réponds : {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # ÉTAPE 3 : Mise à jour du State final
    st.session_state.kernel_state = sol_obj.y[:, -1].tolist()
    st.session_state.last_graph = [sol_obj.y[1].tolist(), sol_obj.y[2].tolist()]
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # ÉTAPE 4 : Rerun propre pour tout redessiner d'un coup
    st.rerun()
