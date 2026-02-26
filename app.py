import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. MOTEUR DE TRANSLATION SÉMANTIQUE ---
# Ce dictionnaire ne contient plus de physique, mais des vecteurs d'argumentation
LOGIQUE_ARGUMENTAIRE = {
    "STRUCTURE": {
        "TECH": "Analyse des fondations : Pourquoi cette idée repose sur des bases fragiles ou solides.",
        "PHILOSOPHIE": "L'héritage conceptuel : D'où vient cette pensée et quel est son ancrage historique.",
        "SOCIÉTÉ": "Le cadre institutionnel : Les règles et les limites du système actuel."
    },
    "DYNAMIQUE": {
        "TECH": "Le levier de croissance : Comment transformer cette base en une action concrète.",
        "PHILOSOPHIE": "La dialectique : Confrontation de l'idée avec son contraire pour créer une synthèse.",
        "SOCIÉTÉ": "Le mouvement collectif : Comment l'opinion ou le marché réagit à cette impulsion."
    },
    "RÉSOLUTION": {
        "TECH": "L'efficience pure : Atteindre l'objectif avec le minimum de friction système.",
        "PHILOSOPHIE": "La sagesse de l'équilibre : Trouver le point de repos entre l'être et le devenir.",
        "SOCIÉTÉ": "L'harmonie sociale : Une solution où l'intérêt individuel rejoint le bien commun."
    }
}

class OracleEngine:
    def __init__(self):
        pass

    def detecter_contexte(self, prompt):
        p = prompt.lower()
        if any(w in p for w in ["code", "python", "import", "système"]): return "TECH"
        if any(w in p for w in ["amour", "beauté", "dieu", "sens", "vie"]): return "PHILOSOPHIE"
        return "SOCIÉTÉ"

    def simuler_pensee_profonde(self, prompt):
        # Simulation mathématique TTU en arrière-plan (invisible pour l'utilisateur)
        t = np.linspace(0, 10, 100)
        coherence = 1.0 + (np.sin(len(prompt)/10) * 0.5)
        # Calcul du point de bascule (Ghost)
        ghost = 0.5 + (len(prompt) % 10) / 10
        df = pd.DataFrame({
            "M": 1.0 * np.exp(-t*0.05),
            "C": coherence + 0.2 * np.random.randn(100),
            "D": 0.1 + 0.05 * np.random.randn(100)
        })
        return df, ghost

# --- 2. CONFIGURATION INTERFACE ---
st.set_page_config(page_title="Oracle V7 - Logique Invisible", layout="wide")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

engine = OracleEngine()

with st.sidebar:
    st.title("👁️ Oracle V7")
    st.caption("Physique invisible | Argumentation pure")
    if st.button("Réinitialiser le champ"):
        st.session_state.conversation = []
        st.rerun()
    st.divider()
    st.info("Le système utilise désormais la TTU-MC³ comme structure de raisonnement logique sans citer les variables.")

# --- 3. LOGIQUE DE RÉPONSE ---
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Exprimez une idée, un concept ou une problématique..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # CHAÎNE DE PENSÉE (Pensée "DeepSeek")
        with st.expander("💭 Analyse profonde...", expanded=True):
            contexte = engine.detecter_contexte(prompt)
            st.write(f"Alignement contextuel : {contexte}...")
            df, g = engine.simuler_pensee_profonde(prompt)
            time.sleep(0.5)
            st.write("Calcul de la trajectoire logique... Point de bascule identifié.")
            time.sleep(0.5)
            st.write("Épuration des termes techniques. Génération de l'argumentaire.")

        # CONSTRUCTION DE LA RÉPONSE (ARGUMENTÉE ET NUANCÉE)
        # On utilise les résultats mathématiques pour choisir le ton
        score_c = df["C"].mean()
        ton = "Affirmatif" if score_c > 1.2 else "Nuancé"

        # Extraction des piliers sans citer M, C, D
        p_struct = LOGIQUE_ARGUMENTAIRE["STRUCTURE"][contexte]
        p_dyn = LOGIQUE_ARGUMENTAIRE["DYNAMIQUE"][contexte]
        p_res = LOGIQUE_ARGUMENTAIRE["RÉSOLUTION"][contexte]

        reponse_finale = f"""
### Analyse et Perspective : {ton}

Suite à l'examen de votre proposition, voici une démonstration articulée en trois axes :

**1. L'Analyse des Fondations**
{p_struct} 
Dans ce contexte, votre question soulève une problématique de stabilité. Il ne s'agit pas seulement de ce que l'on voit, mais des forces invisibles qui maintiennent l'idée en place. Si l'on retire les artifices, il reste une vérité fondamentale sur laquelle nous devons bâtir.

**2. La Dynamique du Mouvement**
{p_dyn}
L'idée n'est pas statique. Elle possède une force d'expansion. Pour que cette pensée devienne réelle, elle doit entrer en collision avec la réalité. C'est dans ce frottement que naît la véritable valeur. La cohérence ici ne vient pas de l'absence de conflit, mais de la capacité à intégrer la contradiction.

**3. Synthèse et Orientation Finale**
{p_res}
Pour aboutir à une conclusion claire : la voie optimale n'est ni dans la rigidité, ni dans l'agitation. Elle réside dans la capacité à agir avec une telle précision que l'effort disparaît. C'est ici que l'argument prend toute sa force.

**En conclusion :** Votre démarche est validée par sa propre logique interne. Pour aller plus loin, concentrez-vous sur le point où l'argument devient une évidence indiscutable.
"""
        st.write(reponse_finale)
        st.session_state.conversation.append({"role": "assistant", "content": reponse_finale})

        with st.expander("📊 Métriques de Pensée (Propriétaire)"):
            st.line_chart(df)
