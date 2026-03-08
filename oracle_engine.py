import os
import json
import uuid
import datetime
import hashlib
import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from github import Github


# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle.db")
GITHUB_REPO = os.getenv("ORACLE_GITHUB_REPO") or os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("ORACLE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")


# ==========================================
# UTILITAIRES GÉNÉRAUX
# ==========================================
def ensure_folder(path: str = MEMORY_FOLDER) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_syllables(word: str):
    return re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy]|$)", word.lower())


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]s+", text) if s.strip()]


def split_paragraphs(text: str):
    return [p.strip() for p in text.split("

") if p.strip()]


def split_words(text: str):
    return re.findall(r"\bw+\b", text.lower())


def to_base64_bytes(content: bytes) -> str:
    import base64
    return base64.b64encode(content).decode("utf-8")


def push_db_to_github():
    """Pousse le fichier SQLite vers GitHub (si correctement configuré)."""
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return

    if not os.path.exists(DB_PATH):
        return

    with open(DB_PATH, "rb") as f:
        content = f.read()

    b64 = to_base64_bytes(content)
    filename = os.path.basename(DB_PATH)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"message": "oracle memory update", "content": b64}

    # On tente un PUT simple (création ou update)
    try:
        resp = requests.put(url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception:
        # On ne casse pas le moteur si GitHub échoue
        pass


# ==========================================
# CLASSE PRINCIPALE
# ==========================================
class OracleEngine:
    def __init__(self):
        ensure_folder()

        # Modèle d'embedding sémantique
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Connexion SQLite
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_tables()

        # GitHub (optionnel)
        self.repo = None
        if GITHUB_TOKEN and GITHUB_REPO:
            try:
                gh = Github(GITHUB_TOKEN)
                self.repo = gh.get_repo(GITHUB_REPO)
            except Exception:
                self.repo = None

        # Cache mémoire
        self.memory = []  # liste de dicts {id, timestamp, text, embedding, source, ...}
        self.embeddings_matrix = None
        self.concept_index = defaultdict(list)

        # Chargement initial
        self.load_memory_from_db()
        self.build_embedding_matrix()
        self.build_concept_index()

    # -------------------------------------------------
    # INITIALISATION SQLite
    # -------------------------------------------------
    def init_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT,
                timestamp TEXT,
                source TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paragraphs (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                embedding TEXT,
                timestamp TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                para_id TEXT,
                text TEXT,
                embedding TEXT,
                timestamp TEXT,
                FOREIGN KEY(para_id) REFERENCES paragraphs(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS words (
                id TEXT PRIMARY KEY,
                word TEXT,
                frequency INTEGER DEFAULT 1,
                doc_id TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS syllables (
                id TEXT PRIMARY KEY,
                syllable TEXT,
                word_id TEXT,
                FOREIGN KEY(word_id) REFERENCES words(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                char TEXT,
                word_id TEXT,
                FOREIGN KEY(word_id) REFERENCES words(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                concept TEXT,
                ref_id TEXT,
                ref_type TEXT,
                PRIMARY KEY (concept, ref_id)
            )
            """
        )

        self.conn.commit()

    # -------------------------------------------------
    # (DE)SÉRIALISATION EMBEDDINGS
    # -------------------------------------------------
    @staticmethod
    def _serialize_embedding(emb: np.ndarray) -> str:
        return json.dumps(emb.tolist())

    @staticmethod
    def _deserialize_embedding(data: str | None):
        if not data:
            return None
        return np.array(json.loads(data))

    # -------------------------------------------------
    # CHARGEMENT DES MÉMOIRES
    # -------------------------------------------------
    def load_memory_from_db(self):
        cursor = self.conn.cursor()

        # Phrases
        cursor.execute(
            "SELECT id, text, embedding, timestamp, para_id FROM sentences"
        )
        for row in cursor.fetchall():
            emb = self._deserialize_embedding(row[2])
            self.memory.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "embedding": emb,
                    "timestamp": row[3],
                    "source": "sentence",
                    "parent": row[4],
                }
            )

        # Paragraphes
        cursor.execute(
            "SELECT id, text, embedding, timestamp, doc_id FROM paragraphs"
        )
        for row in cursor.fetchall():
            emb = self._deserialize_embedding(row[2])
            self.memory.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "embedding": emb,
                    "timestamp": row[3],
                    "source": "paragraph",
                    "parent": row[4],
                }
            )

    # -------------------------------------------------
    # INDEX VECTORIEL + CONCEPTS
    # -------------------------------------------------
    def build_embedding_matrix(self):
        vectors = [
            m["embedding"]
            for m in self.memory
            if m.get("embedding") is not None
        ]
        if not vectors:
            self.embeddings_matrix = None
            return

        mat = np.vstack(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings_matrix = mat / norms

    @staticmethod
    def extract_concepts(text: str):
        words = re.findall(r"\bw+\b", text.lower())
        stop = {
            "le",
            "la",
            "les",
            "de",
            "des",
            "du",
            "un",
            "une",
            "et",
            "en",
            "dans",
            "est",
            "pour",
            "que",
        }
        concepts = [w for w in words if w not in stop and len(w) > 4]
        return list(set(concepts))

    def build_concept_index(self):
        self.concept_index = defaultdict(list)
        for m in self.memory:
            concepts = self.extract_concepts(m["text"])
            for c in concepts:
                self.concept_index[c].append(m)

    # -------------------------------------------------
    # SEGMENTATION SÉMANTIQUE
    # -------------------------------------------------
    @staticmethod
    def semantic_split(text: str):
        sections = re.split(r"
s*d+s*—|

", text)
        return [s.strip() for s in sections if len(s.strip()) > 200]

    # -------------------------------------------------
    # APPRENTISSAGE
    # -------------------------------------------------
    def _insert_word_level(self, cursor, words, doc_id):
        for w in words:
            w_id = hash_text(w)
            cursor.execute(
                "INSERT OR IGNORE INTO words (id, word, frequency, doc_id) "
                "VALUES (?,?,?,?)",
                (w_id, w, 0, doc_id),
            )
            cursor.execute(
                "UPDATE words SET frequency = frequency + 1 WHERE id = ?",
                (w_id,),
            )

            syllables = split_syllables(w)
            for syl in syllables:
                syl_id = hash_text(syl)
                cursor.execute(
                    "INSERT OR IGNORE INTO syllables "
                    "(id, syllable, word_id) VALUES (?,?,?)",
                    (syl_id, syl, w_id),
                )

            for ch in w:
                ch_id = hash_text(ch)
                cursor.execute(
                    "INSERT OR IGNORE INTO characters "
                    "(id, char, word_id) VALUES (?,?,?)",
                    (ch_id, ch, w_id),
                )

    def learn(self, text: str, source: str = "text", doc_id: str | None = None):
        """Apprend un texte : segmentation, embedding, stockage SQLite et cache mémoire."""
        blocks = self.semantic_split(text)
        if not blocks:
            return 0

        timestamp = datetime.datetime.now().isoformat()

        cursor = self.conn.cursor()

        if doc_id is None:
            doc_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT OR IGNORE INTO documents "
                "(id, name, timestamp, source) VALUES (?,?,?,?)",
                (doc_id, source, timestamp, "direct"),
            )

        for block in blocks:
            embedding = self.model.encode(block)
            mem_id = str(uuid.uuid4())

            self.memory.append(
                {
                    "id": mem_id,
                    "timestamp": timestamp,
                    "text": block,
                    "embedding": embedding,
                    "source": source,
                    "doc_id": doc_id,
                }
            )

            cursor.execute(
                "INSERT INTO paragraphs "
                "(id, doc_id, text, embedding, timestamp) "
                "VALUES (?,?,?,?,?)",
                (
                    mem_id,
                    doc_id,
                    block,
                    self._serialize_embedding(embedding),
                    timestamp,
                ),
            )

            sentences = split_sentences(block)
            for sent in sentences:
                sent_emb = self.model.encode(sent)
                sent_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO sentences "
                    "(id, para_id, text, embedding, timestamp) "
                    "VALUES (?,?,?,?,?)",
                    (
                        sent_id,
                        mem_id,
                        sent,
                        self._serialize_embedding(sent_emb),
                        timestamp,
                    ),
                )

                words = split_words(sent)
                self._insert_word_level(cursor, words, doc_id)

            concepts = self.extract_concepts(block)
            for c in concepts:
                cursor.execute(
                    "INSERT OR IGNORE INTO concepts "
                    "(concept, ref_id, ref_type) VALUES (?,?,?)",
                    (c, mem_id, "paragraph"),
                )

        self.conn.commit()

        self.build_embedding_matrix()
        self.build_concept_index()
        push_db_to_github()

        return len(blocks)

    # -------------------------------------------------
    # EXTRACTION TEXTE DEPUIS FICHIER
    # -------------------------------------------------
    @staticmethod
    def extract_text(file):
        text = ""
        file_type = getattr(file, "type", "") or getattr(file, "content_type", "")
        name = getattr(file, "name", "")

        if file_type == "text/plain" or name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")

        elif file_type == "application/pdf" or name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "
"

        elif "word" in file_type or name.endswith(".docx"):
            doc_obj = docx.Document(file)
            for p in doc_obj.paragraphs:
                text += p.text + "
"

        elif "csv" in file_type or name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_string()

        elif "excel" in file_type or name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
            text = df.to_string()

        return text

    def learn_document(self, file):
        """Apprend à partir d'un objet fichier (upload)."""
        text = self.extract_text(file)
        if not text.strip():
            return 0

        doc_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, name, timestamp, source) "
            "VALUES (?,?,?,?)",
            (doc_id, getattr(file, "name", "uploaded"), timestamp, getattr(file, "type", "")),
        )
        self.conn.commit()

        return self.learn(text, source=getattr(file, "name", "uploaded"), doc_id=doc_id)

    # -------------------------------------------------
    # RECHERCHES
    # -------------------------------------------------
    def vector_search(self, question: str, top_k: int = 5):
        if self.embeddings_matrix is None or not self.memory:
            return []

        q_embed = self.model.encode(question)
        q_norm = np.linalg.norm(q_embed)
        if q_norm == 0:
            return []
        q_embed = q_embed / q_norm

        scores = np.dot(self.embeddings_matrix, q_embed)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [self.memory[idx] for idx in top_idx]

    def concept_search(self, question: str):
        concepts = self.extract_concepts(question)
        results = []
        for c in concepts:
            if c in self.concept_index:
                results.extend(self.concept_index[c])
        return results

    # -------------------------------------------------
    # RAISONNEMENT
    # -------------------------------------------------
    def reason(self, question: str):
        vector_results = self.vector_search(question)
        concept_results = self.concept_search(question)
        combined = vector_results + concept_results

        seen = set()
        texts = []
        for m in combined:
            mid = m["id"]
            if mid not in seen:
                texts.append(m["text"])
                seen.add(mid)

        if not texts:
            return "Aucune connaissance pertinente trouvée."

        return "

".join(texts[:3])

    # -------------------------------------------------
    # STATISTIQUES
    # -------------------------------------------------
    def stats(self):
        cursor = self.conn.cursor()
        souvenirs = cursor.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        nb_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        sources = Counter(m.get("source", "unknown") for m in self.memory)

        return {
            "souvenirs": souvenirs,
            "documents": nb_docs,
            "sources": dict(sources),
        }

    # -------------------------------------------------
    # NETTOYAGE
    # -------------------------------------------------
    def close(self):
        self.conn.close()