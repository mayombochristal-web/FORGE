import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer
from fpdf import FPDF

# --- 1. MOTEUR KERNEL TTU (Intégré pour éviter les imports manquants) ---
class TTUKernel:
    def __init__(self):
        self.params = {'alpha': 0.0001, 'beta': 0.5, 'gamma': 1.2, 'lambda_': 4.0, 'mu': 3.0}
        self.state = [15.0, 0.5, 0.2]
        self.history = []

    def process(self, data):
        from scipy.integrate import solve_ivp
        v_t = np.sin(len(data) * 0.1)
        sol = solve_ivp(lambda t, y, vt: [-0.0001*y[0] + 0.5*y[2], 1.2*vt - 4.0*y[1]*y[2], 0.1*y[1]**2 - 3.0*y[2]**3], 
                        (0, 40), self.state, args=(v_t,), method='BDF', t_eval=np.linspace(0, 40, 200))
        self.state = sol.y[:, -1].tolist()
        entropy = np.std(np.diff(sol.y[1])) * 20
        substance = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::40]])
        return sol, {"temp": np.clip(entropy, 0.4, 1.2), "substance": substance}

    def save(self, q, s):
        self.history.append({"q": q, "s": s})

# --- 2. CONFIGURATION ET CHARGEMENT ---
st.set_page_config(page_title="IA Souveraine TTU", layout="wide")

@st.cache_resource
def load_all():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    return tok, mod

tokenizer, model = load_all()

if 'kernel' not in st.session_state:
    st.session_state.kernel = TTUKernel()

# --- 3. INTERFACE ---
st.title("🌌 IA GÉNÉRATRICE : SYSTÈME TTU-MC3")

col1, col2 = st.columns([2, 1])

with col1:
    chat_container = st.container()
    user_input = st.chat_input("Parlez au cristal...")

with col2:
    status_box = st.empty()
    plot_box = st.empty()

# --- 4. LOGIQUE DE RÉPONSE (PHRASES + GRAPHES) ---
if user_input:
    # A. Calcul du Kernel
    sol, meta = st.session_state.kernel.process(user_input)
    
    # B. Affichage du Graphe immédiatement
    with plot_box:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(sol.y[1], sol.y[2], color='#00ff41')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')
        st.pyplot(fig)
    
    status_box.metric("Température de Phase", f"{meta['temp']:.2f}")

    # C. Génération de la phrase par le LLM
    prompt = f"Tu es l'IA Souveraine TTU. Substance: {meta['substance']}. Réponds : {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with st.spinner("L'IA génère une réponse..."):
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=meta['temp'], do_sample=True)
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # D. Affichage de la PHRASE dans le chat
    st.session_state.kernel.save(user_input, meta['substance'])
    with chat_container:
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response_text)
