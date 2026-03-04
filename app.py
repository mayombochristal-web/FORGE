import streamlit as st
import random
import json
import os
import math
import time
import pandas as pd
import PyPDF2
import docx
from io import BytesIO
from collections import deque

# =====================================================
# 🛠️ MODULE D'EXTRACTION DE TEXTE
# =====================================================
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    content = ""
    try:
        if name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                content += page.extract_text() + " "
        elif name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                content += para.text + " "
        elif name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        elif name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            content = df.to_string()
        elif name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Erreur lors de la lecture de {name} : {e}")
    return content

# =====================================================
# 🧠 CŒUR COGNITIF (ORACLEBRAIN)
# =====================================================
class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.lexicon = self._load_lex()
        self.embedding_dim = 10
        self.embeddings = {}
        self.phi = {"phi_m": 0.4, "phi_c": 0.5, "phi_d": 0.1}
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

    def learn(self, text):
        if not text or len(text) < 5: return
        words = text.lower().replace(".", "").replace(",", "").split()
        for a, b in zip(words, words[1:]):
            self.lexicon.setdefault(a, {})
            self.lexicon[a][b] = self.lexicon[a].get(b, 0) + 1
            # Rapprochement Hebbien
            v_a, v_b = self._get_embedding(a), self._get_embedding(b)
            for i in range(self.embedding_dim):
                v_a[i] += self.learning_rate * (v_b[i] - v_a[i])
        self._save_lex()

    def generate(self, seed=None):
        if not self.lexicon: return "Mémoire vide. Importez des documents."
        if not seed or seed not in self.lexicon:
            seed = random.choice(list(self.lexicon.keys()))
        
        words = [seed]
        for _ in range(15):
            curr = words[-1]
            if curr not in self.lexicon: break
            opts = self.lexicon[curr]
            next_w = random.choices(list(opts.keys()), weights=list(opts.values()))[0]
            words.append(next_w)
            if len(words) > 2 and next_w == words[-2]: break
        return " ".join(words).capitalize() + "."

# =====================================================
# 🖥️ INTERFACE STREAMLIT
# =====================================================
st.set_page_config(page_title="ORACLE Ω-TTU MULTIMODAL", layout="wide")

if "brain" not in st.session_state:
    st.session_state.brain = OracleBrain()
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar : Importation ---
with st.sidebar:
    st.title("📥 Absorption de Données")
    uploaded_files = st.file_uploader("PDF, Word, Excel, CSV, TXT", 
                                    type=['pdf', 'docx', 'csv', 'xlsx', 'txt'], 
                                    accept_multiple_files=True)
    
    if st.button("🚀 Lancer l'Analyse"):
        if uploaded_files:
            total_text = ""
            for f in uploaded_files:
                with st.spinner(f"Analyse de {f.name}..."):
                    total_text += extract_text_from_file(f)
            st.session_state.brain.learn(total_text)
            st.success(f"Apprentissage terminé ! Lexique : {len(st.session_state.brain.lexicon)} mots.")
        else:
            st.warning("Aucun fichier sélectionné.")

    st.write("---")
    st.info("Note sur l'audio : Pour traiter l'audio (MP3/WAV), utilisez une bibliothèque comme SpeechRecognition ou Whisper en local.")

# --- Zone de Chat ---
st.title("🧠 ORACLE Ω-TTU V6.5")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Réponse basée sur le lexique absorbé
    seeds = [w for w in prompt.lower().split() if w in st.session_state.brain.lexicon]
    response = st.session_state.brain.generate(random.choice(seeds) if seeds else None)
    
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.history.append({"role": "assistant", "content": response})

# Visualisation Corrigée (évite le NameError)
with st.expander("📊 Analyse du Graphe Lexical"):
    lex = st.session_state.brain.lexicon
    if lex:
        data = [{"Source": m, "Cible": c, "Poids": p} for m, targets in lex.items() for c, p in targets.items()]
        st.table(pd.DataFrame(data).sort_values(by="Poids", ascending=False).head(10))
