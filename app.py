import streamlit as st
import random
import json
import os
import numpy as np
from github import Github
import PyPDF2
import docx
import pandas as pd
import io

# -------------------------------------------------------------------
# Configuration et persistance GitHub / locale
# -------------------------------------------------------------------
def get_memory_path():
    """Retourne le chemin du fichier mémoire sur GitHub."""
    if "GITHUB_MEMORY_DIR" in st.secrets:
        # Utiliser le répertoire dédié
        return f"{st.secrets['GITHUB_MEMORY_DIR'].rstrip('/')}/memory.json"
    else:
        # Compatibilité avec l'ancien GITHUB_PATH
        return st.secrets.get("GITHUB_PATH", "memory.json")

def load_memory():
    """
    Charge le lexique et les embeddings depuis GitHub ou depuis un fichier local.
    Retourne un dict avec les clés "lexicon" et "embeddings".
    """
    memory = {"lexicon": {}, "embeddings": {}}
    # 1. Essayer de charger depuis GitHub
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(st.secrets["GITHUB_REPO"])
        branch = st.secrets.get("GITHUB_BRANCH", None)
        path = get_memory_path()

        # Récupérer le fichier (en précisant la branche si fournie)
        if branch:
            file = repo.get_contents(path, ref=branch)
        else:
            file = repo.get_contents(path)

        data = json.loads(file.decoded_content.decode())
        if "embeddings" not in data:
            data["embeddings"] = {}
        return data
    except Exception as e:
        # 2. Fallback sur fichier local
        if os.path.exists("memory.json"):
            with open("memory.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if "embeddings" not in data:
                    data["embeddings"] = {}
                return data
        return memory

def save_memory(memory):
    """
    Sauvegarde le lexique et les embeddings en local puis tente la synchro GitHub.
    """
    # Sauvegarde locale
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

    # Synchronisation GitHub
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(st.secrets["GITHUB_REPO"])
        branch = st.secrets.get("GITHUB_BRANCH", None)
        path = get_memory_path()
        commit_message = "🧠 Synchronisation Oracle"

        # Vérifier si le fichier existe déjà
        try:
            if branch:
                file = repo.get_contents(path, ref=branch)
            else:
                file = repo.get_contents(path)
            # Mise à jour
            repo.update_file(file.path, commit_message, json.dumps(memory, indent=2), file.sha, branch=branch)
        except Exception:
            # Le fichier n'existe pas → création
            repo.create_file(path, commit_message, json.dumps(memory, indent=2), branch=branch)
    except Exception:
        pass  # Échec silencieux, la copie locale est déjà sauvegardée

# -------------------------------------------------------------------
# Fonctions d'extraction de texte depuis les fichiers uploadés
# -------------------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string(index=False)

# -------------------------------------------------------------------
# Classe principale de l'Oracle V6.5
# -------------------------------------------------------------------
class OracleV6_5:
    def __init__(self):
        memory = load_memory()
        self.lexicon = memory.get("lexicon", {})
        self.embeddings = memory.get("embeddings", {})
        self.phi = {"c": 0.6, "m": 0.5, "d": 0.3}
        self.embedding_dim = 10
        self.hebbian_lr = 0.1

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_embedding(self, word):
        if word not in self.embeddings:
            vec = np.random.uniform(-1, 1, self.embedding_dim)
            self.embeddings[word] = self._normalize(vec).tolist()
        return np.array(self.embeddings[word])

    def learn(self, text):
        words = text.lower().split()
        if len(words) < 2:
            return

        for i in range(len(words) - 1):
            a, b = words[i], words[i+1]

            # Mise à jour des transitions
            if a not in self.lexicon:
                self.lexicon[a] = {}
            self.lexicon[a][b] = self.lexicon[a].get(b, 0) + 1

            # Mise à jour hebbienne des embeddings
            v_a = self._get_embedding(a)
            v_b = self._get_embedding(b)

            delta = self.hebbian_lr * (v_b - v_a)
            v_a_new = v_a + delta
            v_b_new = v_b - delta

            self.embeddings[a] = self._normalize(v_a_new).tolist()
            self.embeddings[b] = self._normalize(v_b_new).tolist()

        save_memory({"lexicon": self.lexicon, "embeddings": self.embeddings})

    def generate(self, seed=None, phi_c=None, phi_m=None, phi_d=None):
        if not self.lexicon:
            return "En attente de données..."

        if phi_c is None:
            phi_c = self.phi["c"]
        if phi_m is None:
            phi_m = self.phi["m"]

        if seed and seed in self.lexicon:
            curr = seed.lower()
        else:
            curr = random.choice(list(self.lexicon.keys()))

        result = [curr]
        for _ in range(20):
            if curr not in self.lexicon or not self.lexicon[curr]:
                break

            candidates = list(self.lexicon[curr].keys())
            freqs = np.array(list(self.lexicon[curr].values()), dtype=float)

            v_curr = self._get_embedding(curr)
            sims = np.array([np.dot(v_curr, self._get_embedding(c)) for c in candidates])

            scores = np.log(freqs + 1) + phi_m * sims

            if phi_c > 0:
                scores = scores / phi_c
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
            else:
                probs = np.zeros_like(scores)
                probs[np.argmax(scores)] = 1.0

            try:
                next_word = np.random.choice(candidates, p=probs)
            except ValueError:
                next_word = random.choice(candidates)

            result.append(next_word)
            curr = next_word

        return " ".join(result).capitalize() + "."

    def sleep_cycle(self, threshold=1):
        to_delete_words = []
        for word, transitions in self.lexicon.items():
            weak = [w for w, c in transitions.items() if c <= threshold]
            for w in weak:
                del transitions[w]
            if not transitions:
                to_delete_words.append(word)

        for word in to_delete_words:
            del self.lexicon[word]

        words_in_lexicon = set(self.lexicon.keys())
        for targets in self.lexicon.values():
            words_in_lexicon.update(targets.keys())

        orphan_embeddings = [w for w in self.embeddings if w not in words_in_lexicon]
        for w in orphan_embeddings:
            del self.embeddings[w]

        save_memory({"lexicon": self.lexicon, "embeddings": self.embeddings})
        return len(to_delete_words), len(orphan_embeddings)

# -------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------
st.set_page_config(page_title="ORACLE Ω-TTU V6.5", layout="wide")
st.title("🧠 ORACLE Ω-TTU V6.5 – Système Neuro‑Symbolique Hybride")

if "oracle" not in st.session_state:
    st.session_state.oracle = OracleV6_5()

oracle = st.session_state.oracle

with st.sidebar:
    st.header("📥 Importation de documents")
    uploaded_file = st.file_uploader(
        "Choisir un fichier (PDF, DOCX, CSV, TXT)",
        type=["pdf", "docx", "csv", "txt"]
    )
    if uploaded_file and st.button("Analyser et apprendre"):
        with st.spinner("Extraction et apprentissage en cours..."):
            file_bytes = uploaded_file.read()
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file_bytes)
            elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith(".csv"):
                text = extract_text_from_csv(file_bytes)
            else:
                text = extract_text_from_txt(file_bytes)

            if text.strip():
                oracle.learn(text)
                st.success(f"✅ Texte extrait et appris ({len(text.split())} mots).")
            else:
                st.error("Aucun texte exploitable trouvé.")

    st.markdown("---")
    st.header("⚙️ Homéostasie Φ")

    phi_c = st.slider(
        "Φc – Cohérence (déterministe ↔ créatif)",
        min_value=0.0, max_value=1.5, value=oracle.phi["c"], step=0.05,
        help="Température du softmax. 0 = toujours le mot le plus probable, 1.5 = plus d'aléatoire."
    )
    phi_m = st.slider(
        "Φm – Mémoire sémantique (influence des embeddings)",
        min_value=0.0, max_value=2.0, value=oracle.phi["m"], step=0.1,
        help="Poids accordé à la proximité des vecteurs dans le choix du mot suivant."
    )
    phi_d = st.slider(
        "Φd – Dissipation (seuil d'élagage)",
        min_value=0, max_value=5, value=int(oracle.phi["d"]), step=1,
        help="Nombre d'occurrences minimal pour conserver une transition (utilisé par le cycle de sommeil)."
    )

    oracle.phi["c"] = phi_c
    oracle.phi["m"] = phi_m
    oracle.phi["d"] = phi_d

    if st.button("😴 Déclencher le cycle de sommeil (élagage)"):
        with st.spinner("Élagage synaptique en cours..."):
            del_words, del_embs = oracle.sleep_cycle(threshold=int(phi_d))
        st.success(f"Élagage terminé : {del_words} mots retirés, {del_embs} embeddings orphelins supprimés.")

    st.markdown("---")
    st.header("📊 Statistiques mémoire")
    st.write(f"**Mots distincts** : {len(oracle.lexicon)}")
    total_transitions = sum(len(v) for v in oracle.lexicon.values())
    st.write(f"**Transitions enregistrées** : {total_transitions}")
    st.write(f"**Taille des embeddings** : {oracle.embedding_dim}")

    if st.button("🔄 Forcer la sauvegarde"):
        save_memory({"lexicon": oracle.lexicon, "embeddings": oracle.embeddings})
        st.success("Mémoire sauvegardée.")

st.subheader("💬 Dialogue avec l'Oracle")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Votre message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    oracle.learn(prompt)

    with st.spinner("L'Oracle réfléchit..."):
        words = prompt.lower().split()
        seed_word = words[-1] if words else None
        response = oracle.generate(seed=seed_word, phi_c=phi_c, phi_m=phi_m)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

    save_memory({"lexicon": oracle.lexicon, "embeddings": oracle.embeddings})

st.markdown("---")
st.caption("ORACLE Ω-TTU V6.5 – Architecture neuro‑symbolique à mémoire persistante. Les embeddings (dim 10) évoluent par apprentissage hebbien.")