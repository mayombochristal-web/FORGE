import streamlit as st
import random
import json
import os
import math
import time
from collections import deque

# =====================================================
# 🛠️ MODULE D'INJECTION (INSERTS PRÉ-CALCULÉS)
# =====================================================
INSERTS_HISTORIQUES = {
    "oracle": {"est": 5, "analyse": 4, "génère": 3, "pense": 2},
    "logique": {"structure": 4, "interne": 3, "pure": 2, "binaire": 2},
    "données": {"flux": 3, "stockage": 2, "traitement": 4},
    "conscience": {"artificielle": 5, "émergente": 3, "simulée": 2},
    "système": {"oméga": 4, "complexe": 3, "stable": 2},
    "algorithme": {"récursif": 3, "stochastique": 4, "prédictif": 2},
    "résonance": {"sémantique": 5, "spectrale": 4, "harmonique": 2},
    "matrice": {"transition": 4, "calcul": 3, "latente": 3},
    "phi": {"m": 3, "c": 3, "d": 3},
    "stabilité": {"oméga": 5, "équilibre": 3},
    "créer": {"concept": 3, "lien": 4, "réalité": 2}
}

# =====================================================
# 🧠 CŒUR COGNITIF
# =====================================================
class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.lexicon = self._load_lex()
        
        # Injection automatique si le lexique est vide
        if not self.lexicon:
            self.lexicon = INSERTS_HISTORIQUES
            self._save_lex()

        # --- Paramètres Latents ---
        self.embedding_dim = 10
        self.embeddings = {}
        self.phi = {"phi_m": 0.4, "phi_c": 0.5, "phi_d": 0.1}
        self.dialog_memory = deque(maxlen=10)
        self.learning_rate = 0.1

    def _load_lex(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_lex(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.lexicon, f, indent=2, ensure_ascii=False)

    def _get_embedding(self, word):
        if word not in self.embeddings:
            vec = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
            norm = math.sqrt(sum(x*x for x in vec)) or 1
            self.embeddings[word] = [x/norm for x in vec]
        return self.embeddings[word]

    def regulate(self, excitation):
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation * 0.05))
        self.phi["phi_c"] = min(1, max(0.2, self.phi["phi_c"] + 0.02))
        total = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= total

    def _compute_score(self, word, candidate, freq):
        # Similarité Cosinus simplifiée
        vec_a = self._get_embedding(word)
        vec_b = self._get_embedding(candidate)
        sim = sum(a*b for a, b in zip(vec_a, vec_b))
        return freq * (1 + 1.5 * max(0, sim))

    def generate(self, seed):
        if seed not in self.lexicon:
            seed = random.choice(list(self.lexicon.keys())) if self.lexicon else "oracle"
        
        words = [seed]
        target_length = int(8 + self.phi["phi_m"] * 15)
        
        for _ in range(target_length):
            curr = words[-1]
            if curr not in self.lexicon: break
            
            opts = self.lexicon[curr]
            scores = {cand: self._compute_score(curr, cand, f) for cand, f in opts.items()}
            
            # Application de la température via phi_c
            if random.random() < self.phi["phi_c"]:
                next_w = max(scores, key=scores.get)
            else:
                next_w = random.choices(list(scores.keys()), weights=list(scores.values()))[0]
            
            words.append(next_w)
            if len(words) > 2 and next_w == words[-2]: break # Éviter boucles infinies
            
        return " ".join(words).capitalize() + "."

    def learn(self, text):
        words = text.lower().replace(".", "").replace(",", "").split()
        if len(words) < 2: return
        
        for a, b in zip(words, words[1:]):
            self.lexicon.setdefault(a, {})
            self.lexicon[a][b] = self.lexicon[a].get(b, 0) + 1
            
            # Apprentissage Hebbien (rapprochement des vecteurs)
            v_a = self._get_embedding(a)
            v_b = self._get_embedding(b)
            for i in range(self.embedding_dim):
                v_a[i] += self.learning_rate * (v_b[i] - v_a[i])
        self._save_lex()

# =====================================================
# 🖥️ INTERFACE STREAMLIT
# =====================================================
st.set_page_config(page_title="ORACLE Ω-TTU V6.5", layout="wide", page_icon="🧠")

if "brain" not in st.session_state:
    st.session_state.brain = OracleBrain()
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 ORACLE Ω-TTU V6.5")
st.caption("Système Hybride Stochastique & Sémantique")

# --- Barre Latérale : Neuro-Monitoring ---
with st.sidebar:
    st.header("⚡ Constantes Vitales")
    phi = st.session_state.brain.phi
    st.metric("Cohérence (Φc)", f"{phi['phi_c']:.2f}")
    st.metric("Mémoire (Φm)", f"{phi['phi_m']:.2f}")
    st.metric("Dissipation (Φd)", f"{phi['phi_d']:.2f}")
    
    st.write("---")
    st.write(f"📚 Lexique : **{len(st.session_state.brain.lexicon)}** concepts")
    
    if st.button("🗑️ Réinitialiser Mémoire"):
        if os.path.exists("oracle_memory.json"): os.remove("oracle_memory.json")
        st.rerun()

# --- Zone de Dialogue ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Analysez ce concept..."):
    # Affichage User
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Traitement Cerveau
    brain = st.session_state.brain
    brain.learn(prompt)
    brain.regulate(len(prompt)/100)
    
    # Choix de la graine (le mot le plus important du prompt présent dans le lexique)
    potential_seeds = [w for w in prompt.lower().split() if w in brain.lexicon]
    seed = random.choice(potential_seeds) if potential_seeds else None
    
    response = brain.generate(seed)
    
    # Affichage Oracle
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.history.append({"role": "assistant", "content": response})

# --- Visualisation (Tableau de bord) ---
with st.expander("🔍 Analyse des connexions neuronales"):
    # On récupère l'instance depuis la session pour éviter le NameError
    brain_instance = st.session_state.brain 
    
    if brain_instance.lexicon:
        df_lex = []
        for mot, cibles in brain_instance.lexicon.items():
            for cible, poids in cibles.items():
                df_lex.append({"Source": mot, "Cible": cible, "Force": poids})
        
        import pandas as pd
        df = pd.DataFrame(df_lex)
        if not df.empty:
            st.dataframe(df.sort_values(by="Force", ascending=False).head(20), use_container_width=True)
        else:
            st.info("Le lexique est encore trop léger pour l'analyse.")
    else:
        st.info("L'Oracle n'a pas encore de souvenirs.")
