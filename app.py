import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml

from ttu_kernel import TTU_Master_Kernel
from ttu_bridge import TTU_LLM_Bridge
from utils import query_signal, balanced_signal, plot_attractor, plot_time_series

st.set_page_config(page_title="TTU-MC3 Générateur Autonome", page_icon="🌀", layout="wide")

st.title("🌀 TTU-MC3 : Moteur de Cristallisation Informationnelle")
st.markdown("""
Ce simulateur implémente la triade (Mémoire, Cohérence, Dissipation) selon les principes de la TTU-MC3.
Ajustez les paramètres dans la barre latérale pour explorer la dynamique de l'attracteur.
La substance extraite peut être utilisée pour générer du texte via un modèle local (GPT-2).
""")

with st.sidebar:
    st.header("⚙️ Paramètres du Kernel")
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
    duration = st.number_input("Durée (t_max)", value=500, min_value=10, max_value=2000, step=50)
    method = st.selectbox("Méthode d'intégration", ["BDF", "RK45", "LSODA"])

    st.subheader("Signal d'injection")
    signal_type = st.selectbox("Type de signal", ["Aucun", "Query", "Balanced"])
    # Le signal miroir nécessiterait une rétroaction en temps réel, non trivial ici.

    st.subheader("Extraction")
    sampling_rate = st.number_input("Taux d'échantillonnage (pas)", value=500, min_value=10, step=10)

    run_button = st.button("🚀 Lancer la simulation")

params = {
    'alpha': alpha,
    'beta': beta,
    'gamma': gamma,
    'lambda_': lambda_,
    'mu': mu
}
initial_state = [pm0, pc0, pd0]
kernel = TTU_Master_Kernel(params, initial_state)

if run_button:
    with st.spinner("Intégration en cours... (cela peut prendre quelques secondes)"):
        if signal_type == "Query":
            signal_func = lambda t: query_signal(t)
        elif signal_type == "Balanced":
            signal_func = lambda t: balanced_signal(t)
        else:
            signal_func = lambda t: 0.0

        t_span = (0, duration)
        t_eval = np.linspace(0, duration, 10000)
        sol = kernel.run_sequence(t_span, t_eval, signal_func=signal_func, method=method)
        substance = kernel.extract_substance(sampling_rate=sampling_rate)

    st.success("Simulation terminée !")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Attracteur (Φc vs Φd)")
        pc, pd = kernel.get_attractor_data()
        if pc is not None:
            fig = plot_attractor(pc, pd)
            st.pyplot(fig)
            plt.close(fig)
    with col2:
        st.subheader("📊 Séries temporelles")
        t = sol.t
        pm, pc, pd = sol.y
        fig2 = plot_time_series(t, pm, pc, pd)
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("🧬 Substance extraite")
    st.code(substance, language="text")

    bridge = TTU_LLM_Bridge(kernel)
    semantic = bridge.extract_semantic_vector()
    st.subheader("🔮 Paramètres sémantiques pour LLM")
    st.write(f"**Température**: {semantic['temperature']:.4f}")
    st.write(f"**Top_p**: {semantic['top_p']:.4f}")

    prompt = bridge.decode_substance_to_prompt(substance)
    st.write(f"**Prompt fantôme**: {prompt}")

    with st.expander("🤖 Génération de texte avec modèle local (GPT-2)"):
        use_llm = st.checkbox("Utiliser GPT-2 pour générer du texte (nécessite transformers et torch)")
        if use_llm:
            try:
                from transformers import pipeline, set_seed
                @st.cache_resource
                def load_model():
                    return pipeline('text-generation', model='gpt2')
                generator = load_model()
                set_seed(42)

                if st.button("Générer du texte"):
                    with st.spinner("Génération en cours..."):
                        results = generator(
                            prompt,
                            max_length=100,
                            temperature=semantic['temperature'],
                            top_p=semantic['top_p'],
                            do_sample=True,
                            num_return_sequences=1
                        )
                        generated = results[0]['generated_text']
                        st.markdown(f"**Résultat:**\n\n{generated}")
            except ImportError:
                st.error("Les bibliothèques `transformers` et `torch` ne sont pas installées. Installez-les avec `pip install transformers torch`.")

    st.subheader("💾 Sauvegarde / Export")
    config = {
        'engine_parameters': params,
        'initial_state': initial_state,
        'integration_settings': {
            'solver': method,
            't_span': [0, duration],
            'sampling_rate': sampling_rate
        },
        'signal_type': signal_type
    }
    yaml_str = yaml.dump(config)
    st.download_button("📥 Télécharger la configuration (YAML)", data=yaml_str, file_name="ttu_config.yaml")
    st.download_button("📥 Télécharger la substance (TXT)", data=substance, file_name="substance.txt")

else:
    st.info("Ajustez les paramètres et cliquez sur 'Lancer la simulation'.")

st.markdown("---")
st.markdown("**TTU-MC3 - Scellement Isomorphique Certifié**")