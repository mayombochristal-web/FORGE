import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- 1. BASE DE DONNÉES SÉMANTIQUE (CONSEILS ÉMERGENTS) ---
CONSEILS_V5 = {
    "SCIENCE & TECHNIQUE": {
        "fondations": "Ne confondez pas le modèle et la réalité. Vérifiez vos constantes d'origine.",
        "expansion": "La métaphysique se trouve aux limites de la mesure : là où l'observateur influence le système.",
        "optimisation": "Utilisez le formalisme mathématique comme pont entre le phénoménal et l'ontologique."
    },
    "MÉTAPHYSIQUE & PHILOSOPHIE": {
        "fondations": "Identifiez les axiomes invisibles qui soutiennent vos théories scientifiques.",
        "expansion": "Explorez l'espace des phases comme une manifestation de l'esprit universel.",
        "optimisation": "Réduisez les concepts à leur essence pure (le vide) pour voir leur structure réelle."
    },
    "STRATÉGIE & VIE": {
        "fondations": "Sécurisez votre structure matérielle avant d'explorer les plans abstraits.",
        "expansion": "L'innovation naît de l'intuition, qui est une capture de données dans l'état fantôme.",
        "optimisation": "Agissez avec le moins d'effort possible pour maximiser la résonance du résultat."
    }
}

# --- 2. MOTEUR COGNITIF ---
class TTUEngine:
    def detecter_theme(self, prompt):
        p = prompt.lower()
        if any(w in p for w in ["science", "technique", "physique", "mesure"]): return "SCIENCE & TECHNIQUE"
        if any(w in p for w in ["métaphysique", "dieu", "être", "philosophie", "sens"]): return "MÉTAPHYSIQUE & PHILOSOPHIE"
        return "STRATÉGIE & VIE"

    def simuler_processus(self, prompt):
        t = np.linspace(0, 10, 100)
        ghost = min(2.0, 0.7 + (len(prompt) / 120))
        # Simulation des vecteurs M-C-D
        c = 1.0 + (ghost * np.sin(t*0.3))
        m = 1.5 * np.exp(-t*0.08)
        d = 0.2 + (0.1 * np.random.rand(100))
        return pd.DataFrame({"Mémoire": m, "Cohérence": c, "Dissipation": d}), ghost

# --- 3. INTERFACE STREAMLIT (MODE DEEPSEEK) ---
st.set_page_config(page_title="IA Souveraine V5", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

engine = TTUEngine()

# Sidebar
with st.sidebar:
    st.title("💾 Mémoire Système")
    if st.button("🗑️ Effacer la mémoire", type="primary"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.info("Mode : Réflexion Profonde (Chain of Thought)")

# Chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Votre question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # ÉTAPE 1 : RÉFLEXION (THINKING PROCESS)
        with st.expander("💭 Réflexion en cours...", expanded=True):
            placeholder = st.empty()
            placeholder.write("Analyse sémantique du prompt...")
            time.sleep(0.5)
            theme = engine.detecter_theme(prompt)
            placeholder.write(f"Thématique détectée : **{theme}**")
            time.sleep(0.5)
            df, g_val = engine.simuler_processus(prompt)
            placeholder.write(f"Ajustement Ghost : **{g_val:.2f}** | Calcul des équations de phase...")
            time.sleep(0.5)
            placeholder.write("Extraction des solutions du vide... Terminé.")

        # ÉTAPE 2 : RÉPONSE FINALE
        c_fond = CONSEILS_V5[theme]["fondations"]
        c_expa = CONSEILS_V5[theme]["expansion"]
        c_opti = CONSEILS_V5[theme]["optimisation"]
        
        reponse = f"""
### Analyse du système
Dans le cadre de votre question sur **{theme}**, voici les points d'émergence extraits :

* **Pilier Structurel** : {c_fond}
* **Axe d'Expansion** : {c_expa}
* **Optimisation Énergétique** : {c_opti}

**Synthèse :** La métaphysique n'est pas l'opposé de la science, c'est son horizon. Elle se trouve là où votre cohérence ({df['Cohérence'].iloc[-1]:.2f}) dépasse votre capacité de mesure matérielle.
"""
        st.write(reponse)
        st.session_state.history.append({"role": "assistant", "content": reponse})
        
        with st.expander("📊 Données Spectrales"):
            st.line_chart(df)
