import os
import json
import uuid
import datetime
import numpy as np
import re
import pandas as pd
import sqlite3
import hashlib
import requests
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from github import Github
import PyPDF2
import docx

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle.db")
GITHUB_REPO = os.getenv("ORACLE_GITHUB_REPO")
GITHUB_TOKEN = os.getenv("ORACLE_GITHUB_TOKEN")

# ==========================================
# UTILITAIRES
# ==========================================
def ensure_folder():
    if not os.path.exists(MEMORY_FOLDER):
        os.makedirs(MEMORY_FOLDER)

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def split_syllables(word):
    return re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]|$)', word.lower())

def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_words(text):
    return re.findall(r'\b\w+\b', text.lower())

def push_db_to_github():
    """Pousse le fichier SQLite vers GitHub (si configuré)."""
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return
    with open(DB_PATH, "rb") as f:
        content = f.read()
    b64 = content.encode("base64")  # Python 2 style, pour Python 3 utiliser base64.b64encode
    filename = os.path.basename(DB_PATH)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"message": "oracle memory update", "content": b64}
    requests.put(url, json=data, headers=headers)

# ==========================================
# CLASSE PRINCIPALE
# ==========================================
class OracleEngine:
    def __init__(self):
        ensure_folder()
        # Modèle d'embedding sémantique (v1)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Connexion SQLite (v2)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.init_tables()
        
        # GitHub (v1) : pour éventuellement lire des JSON, mais on utilisera surtout le push DB
        try:
            self.github = Github(os.getenv("GITHUB_TOKEN"))
            self.repo = self.github.get_repo(os.getenv("GITHUB_REPO"))
        except:
            self.repo = None
        
        # Cache mémoire (v1)
        self.memory = []          # liste de dicts {id, timestamp, text, embedding, source, ...}
        self.embeddings_matrix = None
        self.concept_index = defaultdict(list)
        
        # Chargement initial depuis SQLite
        self.load_memory_from_db()
        self.build_embedding_matrix()
        self.build_concept_index()

    # -------------------------------------------------
    # INITIALISATION SQLite (v2)
    # -------------------------------------------------
    def init_tables(self):
        cursor = self.conn.cursor()
        # Documents
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            name TEXT,
            timestamp TEXT,
            source TEXT
        )""")
        # Paragraphes
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            embedding BLOB,      # stocké comme JSON ou pickle, ici on mettra du texte
            timestamp TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")
        # Phrases
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id TEXT PRIMARY KEY,
            para_id TEXT,
            text TEXT,
            embedding BLOB,
            timestamp TEXT,
            FOREIGN KEY(para_id) REFERENCES paragraphs(id)
        )""")
        # Mots (avec fréquence)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id TEXT PRIMARY KEY,
            word TEXT,
            frequency INTEGER DEFAULT 1,
            doc_id TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")
        # Syllabes (optionnel)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllables (
            id TEXT PRIMARY KEY,
            syllable TEXT,
            word_id TEXT,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )""")
        # Caractères (optionnel)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            char TEXT,
            word_id TEXT,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )""")
        # Concepts (index inversé)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            concept TEXT,
            ref_id TEXT,         # id de la phrase ou du paragraphe
            ref_type TEXT,       # 'sentence' ou 'paragraph'
            PRIMARY KEY (concept, ref_id)
        )""")
        self.conn.commit()

    # -------------------------------------------------
    # CHARGEMENT DEPUIS SQLite VERS self.memory
    # -------------------------------------------------
    def load_memory_from_db(self):
        cursor = self.conn.cursor()
        # On charge les phrases et les paragraphes comme éléments de mémoire
        # Les phrases
        cursor.execute("SELECT id, text, embedding, timestamp, para_id FROM sentences")
        for row in cursor.fetchall():
            emb = self._deserialize_embedding(row[2])
            self.memory.append({
                "id": row[0],
                "text": row[1],
                "embedding": emb,
                "timestamp": row[3],
                "source": "sentence",
                "parent": row[4]
            })
        # Les paragraphes
        cursor.execute("SELECT id, text, embedding, timestamp, doc_id FROM paragraphs")
        for row in cursor.fetchall():
            emb = self._deserialize_embedding(row[2])
            self.memory.append({
                "id": row[0],
                "text": row[1],
                "embedding": emb,
                "timestamp": row[3],
                "source": "paragraph",
                "parent": row[4]
            })
        # On pourrait aussi charger des documents entiers, mais on préfère les unités fines

    def _serialize_embedding(self, emb):
        # Convertit un numpy array en texte (JSON)
        return json.dumps(emb.tolist())

    def _deserialize_embedding(self, data):
        if data is None:
            return None
        return np.array(json.loads(data))

    # -------------------------------------------------
    # CONSTRUCTION DES INDEX (v1)
    # -------------------------------------------------
    def build_embedding_matrix(self):
        vectors = [m["embedding"] for m in self.memory if "embedding" in m and m["embedding"] is not None]
        if vectors:
            self.embeddings_matrix = np.array(vectors)
            norms = np.linalg.norm(self.embeddings_matrix, axis=1)
            self.embeddings_matrix = self.embeddings_matrix / norms[:, None]
        else:
            self.embeddings_matrix = None

    def extract_concepts(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        stop = {"le","la","les","de","des","du","un","une","et","en","dans","est","pour","que"}
        concepts = [w for w in words if w not in stop and len(w) > 4]
        return list(set(concepts))

    def build_concept_index(self):
        self.concept_index = defaultdict(list)
        for m in self.memory:
            concepts = self.extract_concepts(m["text"])
            for c in concepts:
                self.concept_index[c].append(m)

    # -------------------------------------------------
    # SEGMENTATION SÉMANTIQUE (v1)
    # -------------------------------------------------
    def semantic_split(self, text):
        sections = re.split(r"\n\s*\d+\s*—|\n\n", text)
        return [s.strip() for s in sections if len(s.strip()) > 200]

    # -------------------------------------------------
    # APPRENTISSAGE (v1 + v2)
    # -------------------------------------------------
    def learn(self, text, source="text", doc_id=None):
        """Apprend un texte : segmentation, embedding, stockage SQLite et cache mémoire."""
        blocks = self.semantic_split(text)
        inserted_ids = []
        timestamp = str(datetime.datetime.now())
        
        # Si aucun doc_id fourni, on crée un document factice
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO documents (id, name, timestamp, source) VALUES (?,?,?,?)",
                           (doc_id, source, timestamp, "direct"))
            self.conn.commit()

        for block in blocks:
            embedding = self.model.encode(block)
            mem_id = str(uuid.uuid4())
            
            # Stockage en mémoire cache
            data = {
                "id": mem_id,
                "timestamp": timestamp,
                "text": block,
                "embedding": embedding.tolist(),
                "source": source,
                "doc_id": doc_id
            }
            self.memory.append(data)
            
            # Insertion dans SQLite (comme paragraphe)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO paragraphs (id, doc_id, text, embedding, timestamp) VALUES (?,?,?,?,?)",
                (mem_id, doc_id, block, self._serialize_embedding(embedding), timestamp)
            )
            
            # Découpage en phrases pour stockage fin (optionnel mais recommandé)
            sentences = split_sentences(block)
            for sent in sentences:
                sent_emb = self.model.encode(sent)
                sent_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO sentences (id, para_id, text, embedding, timestamp) VALUES (?,?,?,?,?)",
                    (sent_id, mem_id, sent, self._serialize_embedding(sent_emb), timestamp)
                )
                # Mots / syllabes / caractères (v2)
                words = split_words(sent)
                for w in words:
                    w_id = hash_text(w)
                    cursor.execute(
                        "INSERT OR IGNORE INTO words (id, word, frequency, doc_id) VALUES (?,?,?,?)",
                        (w_id, w, 1, doc_id)
                    )
                    # Incrémente la fréquence si déjà existant
                    cursor.execute("UPDATE words SET frequency = frequency + 1 WHERE id = ?", (w_id,))
                    
                    # Syllabes
                    syllables = split_syllables(w)
                    for syl in syllables:
                        syl_id = hash_text(syl)
                        cursor.execute(
                            "INSERT OR IGNORE INTO syllables (id, syllable, word_id) VALUES (?,?,?)",
                            (syl_id, syl, w_id)
                        )
                    
                    # Caractères
                    for ch in w:
                        ch_id = hash_text(ch)
                        cursor.execute(
                            "INSERT OR IGNORE INTO characters (id, char, word_id) VALUES (?,?,?)",
                            (ch_id, ch, w_id)
                        )
            
            # Concepts pour ce bloc
            concepts = self.extract_concepts(block)
            for c in concepts:
                cursor.execute(
                    "INSERT OR IGNORE INTO concepts (concept, ref_id, ref_type) VALUES (?,?,?)",
                    (c, mem_id, "paragraph")
                )
            
            self.conn.commit()
            inserted_ids.append(mem_id)
        
        # Reconstruction des index
        self.build_embedding_matrix()
        self.build_concept_index()
        
        # Sauvegarde GitHub (push du fichier DB)
        push_db_to_github()
        
        return len(blocks)

    # -------------------------------------------------
    # EXTRACTION TEXTE DEPUIS FICHIER (v1)
    # -------------------------------------------------
    def extract_text(self, file):
        text = ""
        if file.type == "text/plain":
            text = file.read().decode("utf-8")
        elif file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        elif "word" in file.type:
            doc = docx.Document(file)
            for p in doc.paragraphs:
                text += p.text + "\n"
        elif "csv" in file.type:
            df = pd.read_csv(file)
            text = df.to_string()
        elif "excel" in file.type:
            df = pd.read_excel(file)
            text = df.to_string()
        return text

    def learn_document(self, file):
        """Apprend à partir d'un objet fichier (upload)."""
        text = self.extract_text(file)
        # Création d'une entrée document dans SQLite
        doc_id = str(uuid.uuid4())
        timestamp = str(datetime.datetime.now())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, name, timestamp, source) VALUES (?,?,?,?)",
            (doc_id, file.name, timestamp, file.type)
        )
        self.conn.commit()
        return self.learn(text, source=file.name, doc_id=doc_id)

    # -------------------------------------------------
    # RECHERCHES (v1)
    # -------------------------------------------------
    def vector_search(self, question, top_k=5):
        if self.embeddings_matrix is None or len(self.memory) == 0:
            return []
        q_embed = self.model.encode(question)
        q_embed = q_embed / np.linalg.norm(q_embed)
        scores = np.dot(self.embeddings_matrix, q_embed)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.memory[idx] for idx in top_idx]

    def concept_search(self, question):
        concepts = self.extract_concepts(question)
        results = []
        for c in concepts:
            if c in self.concept_index:
                results.extend(self.concept_index[c])
        return results

    # -------------------------------------------------
    # RAISONNEMENT (v1)
    # -------------------------------------------------
    def reason(self, question):
        vector_results = self.vector_search(question)
        concept_results = self.concept_search(question)
        combined = vector_results + concept_results
        seen = set()
        texts = []
        for m in combined:
            if m["id"] not in seen:
                texts.append(m["text"])
                seen.add(m["id"])
        if not texts:
            return "Aucune connaissance pertinente trouvée."
        return "\n\n".join(texts[:3])

    # -------------------------------------------------
    # STATISTIQUES (v1 + v2)
    # -------------------------------------------------
    def stats(self):
        cursor = self.conn.cursor()
        souvenirs = cursor.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        nb_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        # Sources depuis la mémoire cache (pour la diversité)
        sources = Counter([m.get("source", "unknown") for m in self.memory])
        return {
            "souvenirs": souvenirs,
            "documents": nb_docs,
            "sources": dict(sources)
        }

    # -------------------------------------------------
    # NETTOYAGE (optionnel)
    # -------------------------------------------------
    def close(self):
        self.conn.close()