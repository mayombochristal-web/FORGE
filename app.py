import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. MOTEUR D'UNIFICATION COGNITIVE ---
class UnifiedArchitect:
    def __init__(self):
        self.themes = {
            "PHILOSOPHIE": ["amour", "beauté", "conscience", "vie", "sens", "dieu", "âme"],
            "TECHNIQUE": ["code", "python", "système", "mécanique", "ttu", "mc3"],
            "STRATÉGIQUE": ["pouvoir", "entreprise", "succès", "société", "argent"]
        }

    def analyser_contexte(self, prompt):
        p = prompt.lower()
        for theme, keywords in self.themes.items():
            if any(k in p for k in keywords): return theme
        return "GÉNÉRAL"

    def simuler_profondeur(self, prompt, history_len):
        t = np.linspace(0, 10, 100)
        # Le Ghost augmente avec la persistance de la discussion
        ghost = 0.6 + (history_len * 0.08)
        coherence = 1.3 + (ghost * np.sin(t * 0.15))
        df = pd.DataFrame({
            "M": 1.0 * np.exp(-t * 0.05),
            "C": coherence + 0.1 * np.random.randn(100),
            "D": 0.12 * np.exp(-history_len * 0.2) + 0.04 * np.random.randn(100)
        })
        return df, ghost

    def generer_synthese_unique(self, prompt, theme, metrics, history):
        """Fusionne les axes de pensée en une démonstration unique et fluide"""
        c_val = metrics["C"].iloc[-1]
        
        # Récupération du contexte historique
        last_topic = history[-2]["content"] if len(history) > 1 else None
        
        # Construction de l'argumentaire unifié
        if theme == "PHILOSOPHIE":
            base = f"L'approche de '{prompt}' transcende la simple définition pour toucher à la structure même de l'expérience."
        elif theme == "TECHNIQUE":
            base = f"La problématique de '{prompt}' s'inscrit dans une nécessité d'optimisation systémique rigoureuse."
        else:
            base = f"L'analyse de '{prompt}' impose une vision globale des interactions de force en présence."

        # Étayage basé sur l'historique
        if last_topic:
            continuite = f"En prolongeant notre réflexion sur les bases précédemment établies, cette nouvelle étape permet de stabiliser le paradigme."
        else:
            continuite = "Cette réflexion initiale pose les jalons d'une compréhension profonde du sujet."

        # Conclusion de résolution (Point de bascule)
        if c_val > 1.6:
            resolution = "La synthèse finale révèle une convergence absolue : l'argument n'a plus besoin de démonstration tant sa cohérence interne s'impose comme une évidence."
        else:
            resolution = "La résolution actuelle propose un équilibre nuancé, où chaque élément du sujet trouve sa place sans générer de friction conceptuelle."

        return f"{base} {continuite} {resolution}"

# --- 2. INTERFACE STREAMLIT V10 ---
st.set_page_config(page_title="Oracle V10 - L'Unificateur", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

arch = UnifiedArchitect()

with st.sidebar:
    st.title("👁️ Oracle V10")
    st.caption("Mode : Synthèse Unifiée & Résolution Unique")
    if st.button("Réinitialiser la Conscience"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.info("Cette version fusionne Structure, Dynamique et Résolution en un seul bloc argumenté.")

# Affichage du Chat
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Votre sujet de réflexion..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.expander("💭 Distillation du raisonnement (TTU-MC³)...", expanded=True):
            theme = arch.analyser_contexte(prompt)
            df_metrics, g_score = arch.simuler_profondeur(prompt, len(st.session_state.history))
            
            # Génération de la réponse unifiée
            synthese_pure = arch.generer_synthese_unique(
                prompt, theme, df_metrics, st.session_state.history
            )
            time.sleep(0.6)
            st.write(f"Phase : {theme} | Ghost de résolution : {g_score:.2f}")

        # RÉPONSE UNIQUE ET ÉTAYÉE
        reponse_finale = f"""
### 💎 Synthèse & Résolution : {prompt}

{synthese_pure}

---
*Note : Cette résolution est le fruit d'une analyse de phase stabilisée par vos échanges précédents.*
"""
        st.write(reponse_finale)
        st.session_state.history.append({"role": "assistant", "content": reponse_finale})

        with st.expander("📊 Signature de Phase (Backend Logique)"):
            st.line_chart(df_metrics)
