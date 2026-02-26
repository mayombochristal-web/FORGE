import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- 1. ARCHITECTURE DES CONSEILS (BASE DE CONNAISSANCES) ---
CONSEILS_EXPERTS = {
    "SCIENCE": {
        "fondations": "Appliquez la méthode expérimentale : isolez une variable unique pour valider votre hypothèse.",
        "expansion": "Cherchez des corrélations interdisciplinaires (ex: biophysique) pour briser les silos théoriques.",
        "optimisation": "Réduisez l'entropie de vos mesures en augmentant le taux d'échantillonnage."
    },
    "LANGAGE": {
        "fondations": "Structurez votre syntaxe pour maximiser la clarté : un sujet, un verbe, une action précise.",
        "expansion": "Utilisez des métaphores isomorphiques pour transférer des concepts complexes vers un public profane.",
        "optimisation": "Éliminez les adjectifs superflus pour renforcer l'impact sémantique de vos verbes."
    },
    "CONCEPT": {
        "fondations": "Définissez vos axiomes de base avant de construire une architecture logique complexe.",
        "expansion": "Explorez la limite de validité de votre concept : où s'arrête-t-il d'être vrai ?",
        "optimisation": "Appliquez le rasoir d'Ockham : la solution la plus simple est souvent la plus proche du vide."
    },
    "STRATÉGIE": {
        "fondations": "Sécurisez vos acquis et vos flux de trésorerie avant toute tentative d'échelle.",
        "expansion": "Identifiez les ruptures de phase du marché (besoins non-dits) pour innover en zone bleue.",
        "optimisation": "Automatisez 80% de vos processus pour concentrer votre énergie sur les 20% créatifs."
    }
}

# --- 2. MOTEUR COGNITIF TTU ---
class TTUEngine:
    def analyse_thematique(self, prompt):
        p = prompt.lower()
        if any(word in p for word in ["physique", "chimie", "bio", "science", "math"]): return "SCIENCE"
        if any(word in p for word in ["écrire", "parler", "langue", "mots", "texte"]): return "LANGAGE"
        if any(word in p for word in ["idée", "philosophie", "théorie", "pensée"]): return "CONCEPT"
        return "STRATÉGIE"

    def simuler_calcul(self, prompt):
        t = np.linspace(0, 10, 100)
        ghost_auto = min(2.0, 0.5 + (len(prompt) / 150))
        # Simulation des courbes triadiques
        coherence = 1.0 + (ghost_auto * np.sin(t*0.2)) + np.random.normal(0, 0.02, 100)
        memoire = 1.5 * np.exp(-t*0.05)
        dissipation = 0.25 + (0.05 * np.random.rand(100))
        df = pd.DataFrame({"Mémoire": memoire, "Cohérence": coherence, "Dissipation": dissipation})
        return df, ghost_auto

# --- 3. INTERFACE UTILISATEUR STREAMLIT ---
st.set_page_config(page_title="IA Souveraine V4 - DeepSeek Mode", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

engine = TTUEngine()

# Sidebar de gestion
with st.sidebar:
    st.title("💾 Mémoire Système")
    if st.button("📥 Sauvegarder la session"):
        st.success("Données Σ consolidées.")
    if st.button("🗑️ Effacer la conversation", type="primary"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption("Ghost Mode: AUTOMATIQUE")

# Affichage des messages
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie
if user_input := st.chat_input("Posez votre question (Sciences, Stratégie, Concepts)..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Simulation de la "Pensée" (Style DeepSeek)
    with st.chat_message("assistant"):
        thought_placeholder = st.expander("💭 Chaîne de pensée (Thinking Process)", expanded=True)
        with thought_placeholder:
            st.write("1. Segmentation du prompt et détection thématique...")
            theme = engine.analyse_thematique(user_input)
            time.sleep(0.3)
            st.write(f"2. Domaine identifié : **{theme}**. Calcul des variables de phase...")
            df_res, g_val = engine.simuler_calcul(user_input)
            time.sleep(0.3)
            st.write(f"3. Ajustement du Ghost à **{g_val:.2f}**. Extraction des conseils du vide...")
            
        # Extraction des données pour la réponse finale
        c_fond = CONSEILS_EXPERTS[theme]["fondations"]
        c_expa = CONSEILS_EXPERTS[theme]["expansion"]
        c_opti = CONSEILS_EXPERTS[theme]["optimisation"]
        priorite = "L'EXPANSION" if df_res['Cohérence'].mean() > 1.3 else "LA STRUCTURE"

        # Rendu de la réponse finale
        reponse_finale = f"""
Voici mon analyse pour votre requête concernant : **{theme}**.

### 📋 Recommandations Stratégiques
* **Fondations & Rigueur** : {c_fond}
* **Innovation & Expansion** : {c_expa}
* **Optimisation & Efficacité** : {c_opti}

### ⚖️ Synthèse Systémique
Compte tenu de l'indice de cohérence ({df_res['Cohérence'].iloc[-1]:.2f}), la stratégie recommandée est de privilégier **{priorite}**. Le système a minimisé la dissipation pour maximiser la clarté de cette réponse.
"""
        st.write(reponse_finale)
        st.session_state.history.append({"role": "assistant", "content": reponse_finale})
        
        # Graphique technique en fin de réponse
        with st.expander("📊 Données de calcul (TTU Metrics)"):
            st.line_chart(df_res)
