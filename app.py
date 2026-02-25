import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. NOYAU TTU INTÉGRÉ ---
class TTUKernel:
    def __init__(self):
        self.state = [15.0, 0.5, 0.2]
    
    def process(self, data):
        from scipy.integrate import solve_ivp
        v_t = np.sin(len(data) * 0.1)
        def system(t, y):
            pm, pc, pd = y
            return [-0.0001*pm + 0.5*pd, 1.2*v_t - 4.0*pc*pd, 0.1*pc**2 - 3.0*pd**3]
        sol = solve_ivp(system, (0, 40), self.state, method='BDF', t_eval=np.linspace(0, 40, 100))
        self.state = sol.y[:, -1].tolist()
        entropy = np.std(np.diff(sol.y[1])) * 20
        substance = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::20]])
        return sol, {"temp": np.clip(entropy, 0.4, 1.2), "substance": substance}

# --- 2. CONFIGURATION ET CHARGEMENT ---
st.set_page_config(page_title="IA SOUVERAINE", layout="wide")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'kernel' not in st.session_state:
    st.session_state.kernel = TTUKernel()
if 'current_sol' not in st.session_state:
    st.session_state.current_sol = None

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

# --- 3. INTERFACE ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")

col1, col2 = st.columns([2, 1])

# A. AFFICHAGE DE L'HISTORIQUE (COLONNE GAUCHE)
with col1:
    container = st.container()
    for m in st.session_state.messages:
        with container.chat_message(m["role"]):
            st.markdown(m["content"])

# B. MONITEUR (COLONNE DROITE)
with col2:
    st.subheader("État du Cristal")
    plot_placeholder = st.empty()
    if st.session_state.current_sol is not None:
        sol = st.session_state.current_sol
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(sol[0], sol[1], color='#00ff41', lw=0.7)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        plot_placeholder.pyplot(fig)

# --- 4. LOGIQUE DE SAISIE ---
user_input = st.chat_input("Parlez au cristal...")

if user_input:
    # 1. Sauvegarder et afficher immédiatement le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Calculs (LLM + Kernel)
    tokenizer, model = load_llm()
    sol_obj, meta = st.session_state.kernel.process(user_input)
    
    # Stocker le graphe pour le prochain rendu
    st.session_state.current_sol = [sol_obj.y[1], sol_obj.y[2]]
    
    prompt = f"Tu es l'IA Souveraine TTU. Substance: {meta['substance']}. Réponds : {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with st.spinner("Résonance en cours..."):
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=meta['temp'], do_sample=True)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # 3. Sauvegarder la réponse
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 4. REDÉMARRAGE PROPRE
    st.rerun()
