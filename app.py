import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os

# ================================================= =
# 1. ARCHITECTE COGNITIF & DYNAMIQUE DE PHASE (TTU-MC³)
# ================================================= =
class OracleV12Engine:
    def __init__(self):
        # Espace latent : Paramètres que l'IA ajuste "en direct" sur vous
        if "latent_params" not in st.session_state:
            st.session_state.latent_params = {"profondeur": 1.0, "coherence_cible": 1.5, "agilite": 1.0}
        
        self.themes = {
            "PHILOSOPHIE": ["conscience", "vie", "sens", "beauté", "amour", "dieu"],
            "TECHNIQUE": ["code", "ttu", "mc3", "système", "physique", "math"],
            "STRATÉGIQUE": ["pouvoir", "entreprise", "société", "guerre", "argent"]
        }

    def analyser_phase(self, prompt, history_len):
        p = prompt.lower()
        theme = "GÉNÉRAL"
        for t, keywords in self.themes.items():
            if any(k in p for k in keywords): theme = t
        
        # Simulation de la métrique de phase
        t_axis = np.linspace(0, 10, 100)
        # Le Ghost (point de bascule) s'affine avec l'apprentissage latent
        ghost = 0.6 + (history_len * 0.05 * st.session_state.latent_params["profondeur"])
        
        metrics = pd.DataFrame({
            "M (Mémoire)": 1.0 * np.exp(-t_axis * 0.05),
            "C (Cohérence)": (1.3 + ghost * np.sin(t_axis * 0.15)) * st.session_state.latent_params["coherence_cible"],
            "D (Dissipation)": 0.15 * np.exp(-history_len * 0.2) + 0.05 * np.random.randn(100)
        })
        return theme, metrics, ghost

# ================================================= =
# 2. LOGIQUE D'AUTO-AMÉLIORATION & RÉCOMPENSE
# ================================================= =
def enregistrer_rlhf(prompt, initial, refined, reward_score):
    """Enregistre les données pour le futur Fine-Tuning (Apprentissage Profond)"""
    log_file = "oracle_rlhf_data.jsonl"
    entry = {
        "timestamp": time.time(),
        "prompt": prompt,
        "chosen": refined if reward_score >= 3 else initial,
        "rejected": initial if reward_score >= 3 else refined,
        "score": reward_score,
        "latent_state": st.session_state.latent_params
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def appliquer_reward(score):
    """Met à jour l'espace latent (Apprentissage immédiat)"""
    # Si le score est bon, on augmente la profondeur et la cohérence
    facteur = (score - 3) * 0.1
    st.session_state.latent_params["profondeur"] += facteur
    st.session_state.latent_params["coherence_cible"] += facteur * 0.5
    # Bornage de sécurité
    for k in st.session_state.latent_params:
        st.session_state.latent_params[k] = np.clip(st.session_state.latent_params[k], 0.5, 3.0)

# ================================================= =
# 3. INTERFACE DE GÉNÉRATION AUGMENTÉE
# ================================================= =
st.set_page_config(page_title="Oracle V12 - Auto-Amélioration", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

engine = OracleV12Engine()

with st.sidebar:
    st.title("👁️ Oracle V12")
    st.write("**Espace Latent (Appris) :**")
    st.json(st.session_state.latent_params)
    
    if st.button("Réinitialiser l'Apprentissage"):
        st.session_state.history = []
        del st.session_state.latent_params
        st.rerun()
    
    st.divider()
    st.info("Cette version apprend de vos notes (Reward) et génère des logs RLHF pour un futur ré-entraînement.")

# Affichage de la discussion
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

if prompt := st.chat_input("Posez votre question... l'IA va s'auto-corriger"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # ÉTAPE 1 : Analyse de Phase
        theme, metrics, ghost = engine.analyser_phase(prompt, len(st.session_state.history))
        
        with st.expander("💭 Phase de Distillation & Auto-Critique", expanded=True):
            st.write(f"Alignement : **{theme}** | Ghost : {ghost:.2f}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("🛠️ **Génération Initiale...**")
                raw_reply = f"Analyse préliminaire de {prompt}. La structure est stable mais manque de profondeur systémique."
                time.sleep(0.5)
                st.write("✓ Complétée.")
            with col2:
                st.write("🧠 **Auto-Correction & Raffinement...**")
                refined_reply = f"Résolution étayée de {prompt} : En intégrant la dynamique de {theme.lower()}, nous observons un point de bascule où la théorie rencontre l'expérience pure. C'est ici que la cohérence devient souveraine."
                time.sleep(0.8)
                st.write("✓ Améliorée.")
            st.line_chart(metrics)

        # ÉTAPE 2 : Affichage de la Résolution Unique (Fusionnée)
        reponse_finale = f"""
### 💎 Résolution Augmentée : {prompt}

{refined_reply}

---
*Perspective TTU-MC³ : Cohérence calculée à {metrics['C (Cohérence)'].iloc[-1]:.2f}.*
"""
        st.write(reponse_finale)
        st.session_state.history.append({"role": "assistant", "content": reponse_finale})

        # ÉTAPE 3 : Système de Reward (Le levier d'apprentissage)
        st.markdown("---")
        st.write("⭐ **Évaluez cette résolution pour l'apprentissage de l'IA :**")
        score = st.feedback("stars", key=f"score_{len(st.session_state.history)}")
        
        if score is not None:
            # On ajoute +1 car st.feedback commence à 0
            real_score = score + 1
            appliquer_reward(real_score)
            enregistrer_rlhf(prompt, raw_reply, refined_reply, real_score)
            st.toast(f"Apprentissage mis à jour : Profondeur désormais à {st.session_state.latent_params['profondeur']:.2f}")
