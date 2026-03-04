import streamlit as st
import random
import json
import os
import math
import time
from collections import deque, Counter

# =====================================================
# 🧠 ORACLE V6.5 — CŒUR COGNITIF PROFOND
# =====================================================

class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.lexicon = self._load_lex()
        self.nexus_lexicon = {}

        # --- Paramètres du Masque ---  
        self.latent_space = {  
            "DATA": ["structure", "données", "json", "flux", "code"],  
            "MIND": ["conscience", "oracle", "pensée", "logique", "analyse"],  
            "ACTION": ["créer", "construire", "générer", "lier", "forge"]  
        }  

        # --- Homéostasie ---  
        self.phi = {"phi_m": 0.4, "phi_c": 0.5, "phi_d": 0.1}  
        self.dialog_memory = deque(maxlen=60)  
        self.hippocampus = []  
        self.last_sleep = time.time()  

        # --- Deep Learning (Simulé) ---  
        self.embedding_dim = 10  
        self.embeddings = {} # mot -> vecteur  
        self.learning_rate = 0.1  
        self.connecteurs = ["cependant", "néanmoins", "ainsi", "donc", "par ailleurs"]

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

    def attend(self):  
        if not self.dialog_memory and not self.lexicon: return "oracle"
        if not self.dialog_memory: return random.choice(list(self.lexicon.keys()))
        
        # Sélection du mot graine par similarité contextuelle
        last_msg = list(self.dialog_memory)[-1].lower().split()
        candidates = [w for w in last_msg if w in self.lexicon]
        return random.choice(candidates) if candidates else "oracle"

    def regulate(self, excitation):  
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation*0.1))  
        self.phi["phi_c"] = min(1, max(0.3, self.phi["phi_c"] + 0.05))  
        total = sum(self.phi.values())  
        for k in self.phi: self.phi[k] /= total  

    def _compute_score(self, word, candidate, freq):  
        curr_emb = self._get_embedding(word)  
        cand_emb = self._get_embedding(candidate)  
        sim = sum(c*e for c, e in zip(curr_emb, cand_emb))  
        boost = 1.5 if any(candidate in mots for mots in self.latent_space.values()) else 1.0  
        return freq * (1 + 2.0 * max(0, sim)) * boost  

    def generate(self, seed):  
        if not self.lexicon: return "Base de données vide. Parlez-moi pour m'instruire."  
        
        words = [seed]
        length = int(10 + self.phi["phi_m"] * 20)  
        
        for _ in range(length):
            current = words[-1]
            if current not in self.lexicon: break
            
            opts = self.lexicon[current]
            scores = {cand: self._compute_score(current, cand, f) for cand, f in opts.items()}
            
            if not scores: break
            # Choix basé sur la cohérence (phi_c)
            if random.random() < self.phi["phi_c"]:
                next_word = max(scores, key=scores.get)
            else:
                next_word = random.choices(list(scores.keys()), weights=list(scores.values()))[0]
            words.append(next_word)

        return " ".join(words).capitalize() + "."

    def learn(self, text):
        words = text.lower().split()
        for a, b in zip(words, words[1:]):
            self.lexicon.setdefault(a, {})
            self.lexicon[a][b] = self.lexicon[a].get(b, 0) + 1
        self._save_lex()

# =====================================================
# 🖥️ INTERFACE STREAMLIT
# =====================================================

st.set_page_config(page_title="ORACLE Ω-TTU", page_icon="🧠", layout="wide")

# Initialisation du cerveau dans la session
if "oracle" not in st.session_state:
    st.session_state.oracle = OracleBrain()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 ORACLE Ω-TTU V6.5")
st.markdown("---")

# Sidebar : Monitoring des constantes vitales (Φ)
with st.sidebar:
    st.header("📊 État Cognitif (Φ)")
    phi = st.session_state.oracle.phi
    st.progress(phi["phi_m"], text=f"Mémoire (Φm): {phi['phi_m']:.2f}")
    st.progress(phi["phi_c"], text=f"Cohérence (Φc): {phi['phi_c']:.2f}")
    st.progress(phi["phi_d"], text=f"Dissipation (Φd): {phi['phi_d']:.2f}")
    
    if st.button("🌙 Cycle de Sommeil (Optimisation)"):
        with st.spinner("Élagage synaptique..."):
            time.sleep(1)
            st.success("Mémoire consolidée.")

# Zone de Chat
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Envoyez une instruction à l'Oracle..."):
    # 1. Afficher message user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Logique Oracle
    oracle = st.session_state.oracle
    oracle.learn(prompt)
    oracle.dialog_memory.append(prompt)
    oracle.regulate(len(prompt)/100)
    
    seed = oracle.attend()
    response = oracle.generate(seed)

    # 3. Afficher réponse
    time.sleep(0.5) # Simuler réflexion
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Visualisation des données (Optionnel)
with st.expander("📂 Voir le Lexique Interne"):
    st.json(st.session_state.oracle.lexicon)
