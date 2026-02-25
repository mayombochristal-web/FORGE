import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
import hashlib

from ttu_kernel import TTU_Master_Kernel
from ttu_bridge import TTU_LLM_Bridge
from utils import prompt_to_signal, plot_attractor, plot_time_series

# Configuration de la page
st.set_page_config(page_title="TTU-MC3 Chatbot", page_icon="🌀", layout="wide")

st.title("🌀 Chatbot TTU-MC3 : IA générative autonome")
st.markdown("""
Ce chatbot utilise un moteur TTU-MC3 en arrière-plan pour influencer la génération de texte.
Chaque question est transformée en signal qui excite le système dynamique.
L'attracteur obtenu produit des paramètres (température, top_p) qui guident le modèle local (DistilGPT2 par défaut).
""")

# Barre latérale : paramètres
with st.sidebar:
    st.header("⚙️ Paramètres TTU")
    alpha = st.number_input("α (Amortissement mémoire)", value=0.0001, format="%.5f", step=0.0001)
    beta = st.number_input("β (Couplage Dissipation-Mémoire)", value=0.5, format="%.2f", step=0.1)
    gamma = st.number_input("γ (Gain de Cohérence)", value=1.2, format="%.2f", step=0.1)
    lambda_ = st.number_input("λ (Couplage non-linéaire)", value=4.0, format="%.2f", step=0.1)
    mu = st.number_input("μ (Friction cubique)", value=3.0, format="%.2f", step=0.1)

    st.subheader("État initial")
    pm0 = st.number_input("Φm (Mémoire)", value=15.0)
    pc0 = st.number_input("Φc (Cohérence)", value=0.5)
    pd0 = st.number_input("Φd (Dissipation)", value=0.2)

    st.subheader("Simulation")
    t_max = st.number_input("Durée d'intégration (s simulées)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
    n_points = st.number_input("Nombre de points", value=1000, min_value=200, max_value=5000, step=100)
    method = st.selectbox("Méthode d'intégration", ["RK45", "BDF", "LSODA"], index=0)  # RK45 plus rapide

    st.subheader("Modèle de langage")
    model_name = st.selectbox("Modèle", ["distilgpt2", "gpt2", "microsoft/DialoGPT-small"], index=0)
    use_web_noise = st.checkbox("Ajouter du bruit 'web'", value=False)
    repetition_penalty = st.slider("Pénalité de répétition", 1.0, 2.0, 1.2, 0.1)

    st.subheader("Affichage")
    show_attractor = st.checkbox("Afficher l'attracteur", value=False)
    show_params = st.checkbox("Afficher paramètres sémantiques", value=True)

# Initialisation de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache du modèle LLM (chargé une seule fois)
@st.cache_resource
def load_llm(model_name):
    from transformers import pipeline, set_seed
    set_seed(42)
    # Pour les modèles GPT, on spécifie pad_token_id pour éviter les warnings
    return pipeline('text-generation', model=model_name, pad_token_id=50256)

def clean_substance(text):
    """Ne garde que les caractères ASCII imprimables."""
    return ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)

def generate_response(prompt, history, params, initial_state, t_max, n_points, method, model_name, use_web_noise, repetition_penalty):
    kernel = TTU_Master_Kernel(params, initial_state)

    def signal_func(t):
        sig = prompt_to_signal(t, prompt, freq_base=1.0)
        if use_web_noise:
            sig += 0.05 * np.random.normal()
        return sig

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_points)
    sol = kernel.run_sequence(t_span, t_eval, signal_func=signal_func, method=method)

    # Extraction de la substance (échantillonnage adapté)
    sampling_rate = max(1, n_points // 30)
    substance = kernel.extract_substance(sampling_rate=sampling_rate)
    substance_clean = clean_substance(substance)[:100]

    bridge = TTU_LLM_Bridge(kernel)
    semantic = bridge.extract_semantic_vector()

    # Construction du prompt : on utilise les derniers messages (max 4)
    context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    # Prompt plus naturel, sans la substance qui peut être bruitée
    llm_prompt = f"Conversation récente:\n{context}\n\nQuestion: {prompt}\nRéponse:"

    try:
        generator = load_llm(model_name)
        results = generator(
            llm_prompt,
            max_new_tokens=80,  # limite la réponse pour accélérer
            temperature=semantic['temperature'],
            top_p=semantic['top_p'],
            do_sample=True,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            pad_token_id=50256
        )
        reply = results[0]['generated_text'].replace(llm_prompt, "").strip()
        if not reply:
            reply = "Je n'ai pas de réponse pour l'instant."
    except Exception as e:
        reply = f"Désolé, une erreur s'est produite : {e}"

    return reply, sol, substance_clean, semantic

# Interface de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Le cristal TTU entre en résonance...")

        params = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'lambda_': lambda_,
            'mu': mu
        }
        initial_state = [pm0, pc0, pd0]

        reply, sol, substance, semantic = generate_response(
            prompt,
            st.session_state.messages,
            params,
            initial_state,
            t_max,
            n_points,
            method,
            model_name,
            use_web_noise,
            repetition_penalty
        )
        message_placeholder.markdown(reply)

        # Affichage optionnel des paramètres
        if show_params:
            with st.expander("🔮 Paramètres sémantiques extraits"):
                st.write(f"**Température**: {semantic['temperature']:.4f}")
                st.write(f"**Top_p**: {semantic['top_p']:.4f}")
                st.write(f"**Substance (extrait)**: `{substance}`")

        # Affichage optionnel de l'attracteur
        if show_attractor and sol is not None:
            with st.expander("🌀 Attracteur Φc vs Φd"):
                pc, pd = kernel.get_attractor_data()
                if pc is not None:
                    fig = plot_attractor(pc, pd, title="Attracteur pour cette question")
                    st.pyplot(fig)
                    plt.close(fig)

    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("---")
st.markdown("**TTU-MC3 - Scellement Isomorphique Certifié**")