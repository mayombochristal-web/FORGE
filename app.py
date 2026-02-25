import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
import hashlib
import time

# Imports locaux
from ttu_kernel import TTU_Master_Kernel
from ttu_bridge import TTU_LLM_Bridge
from utils import prompt_to_signal, plot_attractor, plot_time_series

# Configuration de la page
st.set_page_config(page_title="TTU-MC3 Chatbot", page_icon="🌀", layout="wide")

# Titre et description
st.title("🌀 Chatbot TTU-MC3 : IA générative autonome")
st.markdown("""
Ce chatbot utilise un moteur TTU-MC3 en arrière-plan pour influencer la génération de texte.
Chaque question est transformée en signal qui excite le système dynamique.
L'attracteur obtenu produit des paramètres (température, top_p) et une "substance" qui guident le modèle local GPT-2.
""")

# Barre latérale : paramètres du kernel et options
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
    t_max = st.number_input("Durée d'intégration (secondes simulées)", value=10.0, min_value=1.0, max_value=50.0, step=1.0)
    n_points = st.number_input("Nombre de points", value=2000, min_value=500, max_value=10000, step=500)
    method = st.selectbox("Méthode d'intégration", ["BDF", "RK45", "LSODA"])

    st.subheader("Modèle de langage")
    model_name = st.selectbox("Modèle", ["gpt2", "distilgpt2", "microsoft/DialoGPT-small"])
    use_web_noise = st.checkbox("Ajouter du bruit 'web' au signal", value=True)

    st.subheader("Affichage")
    show_attractor = st.checkbox("Afficher l'attracteur après chaque réponse", value=False)
    show_params = st.checkbox("Afficher les paramètres sémantiques", value=True)

# Initialisation de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache du modèle LLM (chargé une seule fois)
@st.cache_resource
def load_llm(model_name):
    from transformers import pipeline, set_seed
    set_seed(42)
    return pipeline('text-generation', model=model_name)

# Fonction de génération de réponse
def generate_response(prompt, history, params, initial_state, t_max, n_points, method, model_name, use_web_noise):
    # Créer le kernel avec les paramètres actuels
    kernel = TTU_Master_Kernel(params, initial_state)

    # Construire le signal à partir du prompt + éventuel bruit
    def signal_func(t):
        sig = prompt_to_signal(t, prompt, freq_base=1.0)
        if use_web_noise:
            sig += 0.05 * np.random.normal()
        return sig

    # Intégration
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_points)
    sol = kernel.run_sequence(t_span, t_eval, signal_func=signal_func, method=method)

    # Extraction de la substance (échantillonnage adapté)
    sampling_rate = max(1, n_points // 50)  # environ 50 échantillons
    substance = kernel.extract_substance(sampling_rate=sampling_rate)

    # Paramètres sémantiques
    bridge = TTU_LLM_Bridge(kernel)
    semantic = bridge.extract_semantic_vector()

    # Construction du prompt pour le LLM
    # On inclut les derniers échanges (jusqu'à 6 messages) et la substance
    context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
    llm_prompt = f"{context}\n{substance[:100]}\nUser: {prompt}\nAssistant:"

    # Génération avec le modèle
    try:
        generator = load_llm(model_name)
        results = generator(
            llm_prompt,
            max_length=150,
            temperature=semantic['temperature'],
            top_p=semantic['top_p'],
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=50256  # pour gpt2
        )
        reply = results[0]['generated_text'].replace(llm_prompt, "").strip()
        if not reply:
            reply = "..."
    except Exception as e:
        reply = f"[Erreur du modèle: {e}]"

    return reply, sol, substance, semantic

# Interface de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse
    with st.chat_message("assistant"):
        with st.spinner("Le cristal TTU oscille..."):
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
                use_web_noise
            )
            st.markdown(reply)

            # Affichage optionnel des infos
            if show_params:
                with st.expander("🔮 Paramètres sémantiques extraits"):
                    st.write(f"**Température**: {semantic['temperature']:.4f}")
                    st.write(f"**Top_p**: {semantic['top_p']:.4f}")
                    st.write(f"**Substance (extrait)**: `{substance[:100]}`")

            if show_attractor and sol is not None:
                with st.expander("🌀 Attracteur Φc vs Φd"):
                    pc, pd = kernel.get_attractor_data()
                    if pc is not None:
                        fig = plot_attractor(pc, pd, title="Attracteur pour cette question")
                        st.pyplot(fig)
                        plt.close(fig)

    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Pied de page
st.markdown("---")
st.markdown("**TTU-MC3 - Scellement Isomorphique Certifié**")