import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. KERNEL TTU STABLE ---
class TTUKernel:
    def __init__(self):
        self.state = [15.0, 0.5, 0.2]
        self.history = []

    def process(self, data):
        from scipy.integrate import solve_ivp
        v_t = np.sin(len(data) * 0.1)
        # Équations de phase simplifiées pour la stabilité
        def system(t, y):
            pm, pc, pd = y
            return [-0.0001*pm + 0.5*pd, 1.2*v_t - 4.0*pc*pd, 0.1*pc**2 - 3.0*pd**3]
        
        sol = solve_ivp(system, (0, 40), self.state, method='BDF', t_eval=np.linspace(0, 40, 100))
        self.state = sol.y[:, -1].tolist()
        entropy = np.std(np.diff(sol.y[1])) * 20
        substance = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::20]])
        return sol, {"temp": np.clip(entropy, 0.4, 1.2), "substance": substance}

# --- 2. INITIALISATION SÉCURISÉE ---
st.set_page_config(page_title="IA SOUVERAINE TTU", layout="wide")

if 'kernel' not in st.session_state:
    st.session_state.kernel = TTUKernel()
if 'messages' not in st.session_state:
    st.session_state.messages = [] # On stocke les phrases ici

@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

tokenizer, model = load_model()

# --- 3. INTERFACE ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")

col1, col2 = st.columns([2, 1])

# AFFICHER LES ANCIENNES PHRASES (Historique)
with col1:
    chat_placeholder = st.container()
    for i, msg in enumerate(st.session_state.messages):
        with chat_placeholder.chat_message(msg["role"]):
            st.write(msg["content"])

# ZONE DE SAISIE
user_input = st.chat_input("Parlez au cristal...")

if user_input:
    # On ajoute la phrase de l'utilisateur immédiatement
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Calcul de phase
    sol, meta = st.session_state.kernel.process(user_input)
    
    # Génération LLM
    prompt = f"Tu es l'IA Souveraine TTU. Substance: {meta['substance']}. Réponds brièvement : {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with st.spinner("Résonance..."):
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=meta['temp'], do_sample=True)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Sauvegarde de la phrase de l'IA
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # RELANCE L'INTERFACE POUR TOUT AFFICHER
    st.rerun()

# --- 4. VISUALISATION (COLONNE DROITE) ---
with col2:
    st.subheader("État du Cristal")
    if 'sol' in locals() or 'meta' in locals():
        # On utilise st.empty pour éviter le conflit de noeuds
        plot_area = st.empty()
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(sol.y[1], sol.y[2], color='#00ff41')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        plot_area.pyplot(fig)
        st.metric("Entropie de Phase", f"{meta['temp']:.2f}")
