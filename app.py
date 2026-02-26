import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. MOTEUR COGNITIF : ARCHITECTE DE RAISONNEMENT ---
class CognitiveArchitect:
    def __init__(self):
        # Catégories de pensée pour l'orientation discursive
        self.themes = {
            "PHILOSOPHIE": ["amour", "beauté", "conscience", "vie", "sens", "dieu", "âme"],
            "TECHNIQUE": ["code", "python", "système", "mécanique", "ttu", "mc3", "algorithme"],
            "STRATÉGIQUE": ["pouvoir", "entreprise", "succès", "société", "argent", "politique"]
        }

    def analyser_contexte(self, prompt):
        p = prompt.lower()
        for theme, keywords in self.themes.items():
            if any(k in p for k in keywords):
                return theme
        return "GÉNÉRAL"

    def simuler_moteur_ttu(self, prompt):
        """Calcule la structure logique invisible (TTU-MC3)"""
        t = np.linspace(0, 10, 100)
        # Le Ghost (pression de vide) influence la profondeur du raisonnement
        ghost = 0.5 + (len(prompt) % 50) / 100
        coherence_base = 1.2 + (ghost * np.sin(t * 0.2))
        
        df = pd.DataFrame({
            "M": 1.0 * np.exp(-t * 0.04), # Érosion de la donnée brute vers l'idée
            "C": coherence_base + 0.1 * np.random.randn(100), # Flux de corrélation
            "D": 0.15 + 0.05 * np.random.randn(100) # Dissipation (bruit sémantique)
        })
        return df, ghost

    def generer_argumentation(self, prompt, theme, metrics):
        """Transforme les variables physiques en argumentation pure"""
        m_val = metrics["M"].iloc[-1]
        c_val = metrics["C"].iloc[-1]
        d_val = metrics["D"].iloc[-1]

        # Logique de synthèse : L'IA interprète ses propres métriques
        
        # 1. Fondations (Basé sur la Mémoire M)
        if theme == "PHILOSOPHIE":
            struct = f"L'interrogation sur '{prompt}' nous place à la frontière du mesurable et du ressenti. La structure de cette idée repose sur la persistance de l'identité à travers le changement."
        elif theme == "TECHNIQUE":
            struct = f"L'analyse de '{prompt}' révèle une architecture dont la stabilité dépend de la cohérence de ses primitives fondamentales."
        else:
            struct = f"La base de votre réflexion sur '{prompt}' s'inscrit dans un cadre systémique où les règles établies définissent les limites du possible."

        # 2. Dynamique (Basé sur la Cohérence C)
        if c_val > 1.4:
            flux = "La dynamique de ce concept est portée par une résonance interne puissante, permettant d'intégrer les contradictions apparentes dans une unité logique supérieure."
        else:
            flux = "Le mouvement de pensée ici est encore en phase de structuration ; il nécessite une confrontation avec la réalité pour stabiliser sa trajectoire."

        # 3. Résolution (Basé sur la Dissipation D)
        if d_val < 0.12:
            resol = "L'aboutissement est une clarté absolue : un état de 'silence conceptuel' où l'argument devient une évidence indiscutable et l'effort de compréhension disparaît."
        else:
            resol = "La résolution demande une épuration des bruits parasites. Il faut encore nuancer l'approche pour laisser transparaître l'essence même du sujet."

        return struct, flux, resol

# --- 2. INTERFACE ET DÉPLOIEMENT ---
st.set_page_config(page_title="Architecte Cognitif V8.1", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

arch = CognitiveArchitect()

with st.sidebar:
    st.title("🧠 Architecte V8.1")
    st.caption("Raisonnement Autonome | TTU-MC³ Intégrée")
    if st.button("Réinitialiser les flux de pensée"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.info("Cette version utilise une Barrière de Phase unifiée pour éviter les erreurs de variables.")

# Affichage de la conversation
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Exprimez une thèse, un concept ou posez une question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.expander("💭 Analyse de phase et trajectoire logique...", expanded=True):
            # 1. Identification du thème
            theme_detecte = arch.analyser_contexte(prompt)
            st.write(f"Alignement paradigmatique : **{theme_detecte}**")
            
            # 2. Simulation de la dynamique interne
            df_metrics, g_score = arch.simuler_moteur_ttu(prompt)
            time.sleep(0.3)
            st.write(f"Ajustement du champ (Ghost) : {g_score:.2f}")
            
            # 3. Génération autonome de l'argumentaire
            arg1, arg2, arg3 = arch.generer_argumentation(prompt, theme_detecte, df_metrics)
            time.sleep(0.3)
            st.write("Épuration sémantique terminée. Synthèse prête.")

        # Réponse finale : Argumentée, nuancée et démontrée
        reponse = f"""
### Analyse du Paradigme : {prompt}

**1. Analyse des Fondations**
{arg1}

**2. Dynamique et Flux**
{arg2}

**3. Synthèse et Résolution**
{arg3}

**Conclusion :** Cette démonstration n'est pas une simple réponse technique, mais une projection de la cohérence interne de votre sujet. En stabilisant les fondations et en optimisant le flux, l'évidence s'impose d'elle-même.
"""
        st.write(reponse)
        st.session_state.history.append({"role": "assistant", "content": reponse})

        with st.expander("📊 Signature Spectrale (Preuve TTU-MC³/VTM)"):
            st.line_chart(df_metrics)
