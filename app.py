import streamlit as st
import random
import json
import os
import math
import numpy as np
from collections import defaultdict, deque
from github import Github
import PyPDF2
import docx
import pandas as pd
import io

# -------------------------------------------------------------------
# Configuration et persistance GitHub / locale
# -------------------------------------------------------------------
def load_memory():
    """
    Charge le lexique et les embeddings depuis GitHub ou depuis un fichier local.
    Retourne un dict avec les clés "lexicon" et "embeddings".
    """
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(st.secrets["GITHUB_REPO"])
        file = repo.get_contents(st.secrets["GITHUB_PATH"])
        data = json.loads(file.decoded_content.decode())
        # S'assurer que les embeddings existent (rétrocompatibilité)
        if "embeddings" not in data:
            data["embeddings"] = {}
        return data
    except Exception as e:
        # Fallback sur fichier local
        if os.path.exists("memory.json"):
            with open("memory.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if "embeddings" not in data:
                    data["embeddings"] = {}
                return data
        return {"lexicon": {}, "embeddings": {}}

def save_memory(memory):
    """
    Sauvegarde le lexique et les embeddings en local puis tente la synchro GitHub.
    """
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(st.secrets["GITHUB_REPO"])
        file = repo.get_contents(st.secrets["GITHUB_PATH"])
        repo.update_file(file.path, "🧠 Synchronisation Oracle", json.dumps(memory, indent=2), file.sha)
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
    # Convertir le dataframe en une chaîne lisible (clés + valeurs)
    return df.to_string(index=False)

# -------------------------------------------------------------------
# Classe principale de l'Oracle V6.5
# -------------------------------------------------------------------
class OracleV6_5:
    def __init__(self):
        # Chargement de la mémoire (lexique + embeddings)
        memory = load_memory()
        self.lexicon = memory.get("lexicon", {})
        self.embeddings = memory.get("embeddings", {})
        # Paramètres homéostasiques (valeurs par défaut)
        self.phi = {"c": 0.6, "m": 0.5, "d": 0.3}
        self.embedding_dim = 10
        self.hebbian_lr = 0.1

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_embedding(self, word):
        """Retourne le vecteur d'un mot, en crée un aléatoire normalisé si absent."""
        if word not in self.embeddings:
            # Initialisation aléatoire uniforme puis normalisation
            vec = np.random.uniform(-1, 1, self.embedding_dim)
            self.embeddings[word] = self._normalize(vec).tolist()
        return np.array(self.embeddings[word])

    def learn(self, text):
        """
        Apprentissage supervisé :
        - Met à jour les transitions (bigrammes)
        - Applique la règle hebbienne sur les embeddings des mots consécutifs
        """
        words = text.lower().split()
        if len(words) < 2:
            return

        for i in range(len(words) - 1):
            a, b = words[i], words[i+1]

            # 1. Mise à jour du lexique (compteurs de transition)
            if a not in self.lexicon:
                self.lexicon[a] = {}
            self.lexicon[a][b] = self.lexicon[a].get(b, 0) + 1

            # 2. Mise à jour hebbienne des embeddings
            v_a = self._get_embedding(a)
            v_b = self._get_embedding(b)

            # Règle hebbienne : rapprocher les vecteurs
            delta = self.hebbian_lr * (v_b - v_a)
            v_a_new = v_a + delta
            v_b_new = v_b - delta   # symétrique

            self.embeddings[a] = self._normalize(v_a_new).tolist()
            self.embeddings[b] = self._normalize(v_b_new).tolist()

        # Sauvegarde après apprentissage
        save_memory({"lexicon": self.lexicon, "embeddings": self.embeddings})

    def generate(self, seed=None, phi_c=None, phi_m=None, phi_d=None):
        """
        Génération de texte en combinant probabilités de transition et similarité sémantique.
        - phi_c (cohérence) : température du softmax (0 = déterministe, 1 = créatif)
        - phi_m (mémoire)   : pondération de la similarité embedding (0 = ignore, 1 = forte influence)
        - phi_d (dissipation): non utilisé directement ici, sert pour l'élagage (sleep cycle)
        """
        if not self.lexicon:
            return "En attente de données..."

        # Utilisation des paramètres courants si non fournis
        if phi_c is None:
            phi_c = self.phi["c"]
        if phi_m is None:
            phi_m = self.phi["m"]

        # Choix du mot de départ
        if seed and seed in self.lexicon:
            curr = seed.lower()
        else:
            curr = random.choice(list(self.lexicon.keys()))

        result = [curr]
        for _ in range(20):  # longueur fixe de génération
            if curr not in self.lexicon or not self.lexicon[curr]:
                break

            # Liste des candidats et de leurs fréquences
            candidates = list(self.lexicon[curr].keys())
            freqs = np.array(list(self.lexicon[curr].values()), dtype=float)

            # Calcul des similarités cosinus avec le mot courant
            v_curr = self._get_embedding(curr)
            sims = np.array([np.dot(v_curr, self._get_embedding(c)) for c in candidates])

            # Combinaison : log(freq+1) + phi_m * sim
            # (phi_m contrôle le poids de la similarité)
            scores = np.log(freqs + 1) + phi_m * sims

            # Température (phi_c)
            if phi_c > 0:
                scores = scores / phi_c
                exp_scores = np.exp(scores - np.max(scores))  # stabilité numérique
                probs = exp_scores / np.sum(exp_scores)
            else:
                # Mode déterministe : prendre le meilleur score
                probs = np.zeros_like(scores)
                probs[np.argmax(scores)] = 1.0

            # Choix du prochain mot
            try:
                next_word = np.random.choice(candidates, p=probs)
            except ValueError:
                # En cas de problème de somme, on prend au hasard
                next_word = random.choice(candidates)

            result.append(next_word)
            curr = next_word

        return " ".join(result).capitalize() + "."

    def sleep_cycle(self, threshold=1):
        """
        Élagage synaptique : supprime les transitions trop faibles (compteur <= threshold)
        et les mots isolés.
        """
        # Nettoyage du lexique
        to_delete_words = []
        for word, transitions in self.lexicon.items():
            # Supprimer les transitions faibles
            weak = [w for w, c in transitions.items() if c <= threshold]
            for w in weak:
                del transitions[w]
            # Si le mot n'a plus de transitions, marquer pour suppression
            if not transitions:
                to_delete_words.append(word)

        for word in to_delete_words:
            del self.lexicon[word]
            # On ne supprime pas automatiquement l'embedding, mais on pourra le faire plus tard

        # Option : supprimer les embeddings des mots absents du lexique
        words_in_lexicon = set(self.lexicon.keys())
        # On garde aussi les mots qui apparaissent comme cibles (présents dans les valeurs)
        for targets in self.lexicon.values():
            words_in_lexicon.update(targets.keys())

        orphan_embeddings = [w for w in self.embeddings if w not in words_in_lexicon]
        for w in orphan_embeddings:
            del self.embeddings[w]

        # Sauvegarde après élagage
        save_memory({"lexicon": self.lexicon, "embeddings": self.embeddings})
        return len(to_delete_words), len(orphan_embeddings)

# -------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------
st.set_page_config(page_title="ORACLE Ω-TTU V6.5", layout="wide")
st.title("🧠 ORACLE Ω-TTU V6.5 – Système Neuro‑Symbolique Hybride")

# Initialisation de l'Oracle dans la session
if "oracle" not in st.session_state:
    st.session_state.oracle = OracleV6_5()

oracle = st.session_state.oracle

# Barre latérale : importation et contrôle
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
            else:  # texte brut
                text = extract_text_from_txt(file_bytes)

            if text.strip():
                oracle.learn(text)
                st.success(f"✅ Texte extrait et appris ({len(text.split())} mots).")
            else:
                st.error("Aucun texte exploitable trouvé.")

    st.markdown("---")
    st.header("⚙️ Homéostasie Φ")

    # Φc : cohérence (température)
    phi_c = st.slider(
        "Φc – Cohérence (déterministe ↔ créatif)",
        min_value=0.0, max_value=1.5, value=oracle.phi["c"], step=0.05,
        help="Température du softmax. 0 = toujours le mot le plus probable, 1.5 = plus d'aléatoire."
    )
    # Φm : mémoire (poids de la similarité sémantique)
    phi_m = st.slider(
        "Φm – Mémoire sémantique (influence des embeddings)",
        min_value=0.0, max_value=2.0, value=oracle.phi["m"], step=0.1,
        help="Poids accordé à la proximité des vecteurs dans le choix du mot suivant."
    )
    # Φd : dissipation (seuil d'élagage)
    phi_d = st.slider(
        "Φd – Dissipation (seuil d'élagage)",
        min_value=0, max_value=5, value=int(oracle.phi["d"]), step=1,
        help="Nombre d'occurrences minimal pour conserver une transition (utilisé par le cycle de sommeil)."
    )

    # Mise à jour des paramètres dans l'instance
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

# Zone principale : chat
st.subheader("💬 Dialogue avec l'Oracle")
# Afficher les messages précédents
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Champ de saisie
if prompt := st.chat_input("Votre message..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Apprentissage à partir du prompt
    oracle.learn(prompt)

    # Génération de la réponse
    with st.spinner("L'Oracle réfléchit..."):
        # Extraire un éventuel mot‑graine (le dernier mot du prompt)
        words = prompt.lower().split()
        seed_word = words[-1] if words else None
        response = oracle.generate(seed=seed_word, phi_c=phi_c, phi_m=phi_m)

    # Afficher la réponse
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

    # Sauvegarde automatique après chaque échange (optionnel)
    save_memory({"lexicon": oracle.lexicon, "embeddings": oracle.embeddings})

# Pied de page
st.markdown("---")
st.caption("ORACLE Ω-TTU V6.5 – Architecture neuro‑symbolique à mémoire persistante. Les embeddings (dim 10) évoluent par apprentissage hebbien.")