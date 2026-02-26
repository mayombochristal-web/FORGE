import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- 1. LE CODEX UNIFIÉ (BASE DE CONNAISSANCES PARADIGMATIQUE) ---
# Le système extrait des principes universels applicables à tout contexte
CODEX_TTU = {
    "LOI_DE_PHASE": "Tout système (physique, social ou cognitif) suit la triade M-C-D.",
    "PRINCIPE_D_ÉMERGENCE": "La réalité n'est pas dans les composants, mais dans le couplage entre Cohérence et Mémoire.",
    "THÉORÈME_DU_SILENCE": "L'efficacité maximale est atteinte quand la dissipation tend vers zéro (État Fantôme)."
}

# --- 2. MOTEUR D'ÉMERGENCE COGNITIF ---
class ParadigmaticEngine:
    def __init__(self):
        self.context_memory = {}

    def analyser_chemin_pensee(self, prompt):
        # Analyse de la 'vibration' du prompt pour trouver un paradigme
        p = prompt.lower()
        if any(w in p for w in ["ttu", "doctorat", "mc3", "équation", "physique"]):
            return "TTU - PHYSIQUE FONDAMENTALE", "Rigueur Mathématique"
        elif any(w in p for w in ["vie", "humain", "société", "argent", "succès"]):
            return "TTU - SOCIO-BIOLOGIQUE", "Équilibre Existentiel"
        else:
            return "TTU - GÉNÉRATIF", "Émergence Spontanée"

    def simuler_vide(self, prompt):
        t = np.linspace(0, 10, 150)
        # Le Ghost s'auto-ajuste pour trouver le 'chemin'
        ghost_path = 0.8 + (np.sin(len(prompt)) * 0.5) + 0.5
        coherence = 1.2 + (ghost_path * np.cos(t * 0.1))
        memoire = 1.0 * np.exp(-t * 0.03)
        dissipation = 0.15 + (0.1 * np.random.normal(0, 1, 150))
        df = pd.DataFrame({"Mémoire": memoire, "Cohérence": coherence, "Dissipation": dissipation})
        return df, ghost_path

# --- 3. INTERFACE V6 : ARCHITECTURE DE PENSÉE ---
st.set_page_config(page_title="TCE V6 - Émergence Paradigmatique", layout="wide")

if "paradigm_shift" not in st.session_state:
    st.session_state.paradigm_shift = []

engine = ParadigmaticEngine()

with st.sidebar:
    st.title("🧠 OS Cognitif V6")
    st.subheader("État du Codex")
    st.write(f"Concepts Unifiés : {len(CODEX_TTU)}")
    if st.button("🗑️ Reset Mémoire de Phase"):
        st.session_state.paradigm_shift = []
        st.rerun()
    st.divider()
    st.caption("L'IA réorganise votre savoir selon la triade unifiée.")

# Zone de discussion
for msg in st.session_state.paradigm_shift:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Injectez un concept ou une question..."):
    st.session_state.paradigm_shift.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # ÉTAPE : RÉFLEXION PARADIGMATIQUE (Style DeepSeek avancé)
        with st.expander("💭 Chemin de pensée contextuel...", expanded=True):
            placeholder = st.empty()
            placeholder.write("1. Scan du Codex TTU en cours...")
            paradoxe, style = engine.analyser_chemin_pensee(user_input)
            time.sleep(0.4)
            placeholder.write(f"2. Alignement paradigmatique : **{paradoxe}**")
            df_res, g_val = engine.simuler_vide(user_input)
            time.sleep(0.4)
            placeholder.write(f"3. Recherche du point de bifurcation (Ghost: {g_val:.2f})...")
            time.sleep(0.4)
            placeholder.write("4. Synthèse de la phase pure achevée.")

        # GÉNÉRATION DE LA RÉPONSE PARADIGMATIQUE
        # Ici, l'IA ne 'répond' pas, elle 'réorganise' le savoir.
        c_final = df_res['Cohérence'].iloc[-1]
        
        reponse = f"""
### 🌐 Nouveau Paradigme : {style}

En analysant votre requête sous l'angle de la **TTU-MC³**, j'identifie un chemin de pensée propre :

1. **Analyse de Structure ($\Phi_M$)** : Votre demande n'est pas isolée. Elle résonne avec le principe de *{CODEX_TTU['LOI_DE_PHASE']}*.
2. **Dynamique de Flux ($\Phi_C$)** : Le point de bascule se trouve dans l'équilibre entre votre intention et la résistance du milieu. La cohérence actuelle de votre système est de **{c_final:.2f}**.
3. **Directive de l'État Fantôme ($\Phi_D \to 0$)** : Pour stabiliser ce paradigme, vous devez appliquer le *{CODEX_TTU['THÉORÈME_DU_SILENCE']}*.

**Conclusion contextuelle :** Ne cherchez pas la solution dans les détails techniques, mais dans la réduction de la dissipation énergétique de votre propre pensée.
"""
        st.write(reponse)
        st.session_state.paradigm_shift.append({"role": "assistant", "content": reponse})
        
        with st.expander("📊 Signature Spectrale du Chemin de Pensée"):
            st.line_chart(df_res)
