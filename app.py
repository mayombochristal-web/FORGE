import streamlit as st
import os
import json
import uuid
import datetime
import hashlib
import re
import sqlite3
import tempfile
import base64
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

# ==========================================
# UTILITAIRES
# ==========================================
def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def split_syllables(word):
    """Découpe un mot en syllabes (approximatif)."""
    return re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]|$)', word.lower())

def split_sentences(text):
    """Découpe un texte en phrases."""
    return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

def split_paragraphs(text):
    """Découpe un texte en paragraphes."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_words(text):
    """Extrait les mots d'un texte."""
    return re.findall(r'\b\w+\b', text.lower())

# ==========================================
# CLASSE PRINCIPALE ORACLE ENGINE
# ==========================================
class OracleEngine:
    def __init__(self):
        # Modèle d'embedding sémantique
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Connexion SQLite
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_tables()
        
        # Cache mémoire
        self.memory = []          # liste de dicts
        self.embeddings_matrix = None
        self.concept_index = defaultdict(list)
        
        # Chargement initial depuis SQLite
        self.load_memory_from_db()
        self.build_embedding_matrix()
        self.build_concept_index()

    # -------------------------------------------------
    # INITIALISATION SQLite
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
            embedding TEXT,
            timestamp TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")
        # Phrases
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id TEXT PRIMARY KEY,
            para_id TEXT,
            text TEXT,
            embedding TEXT,
            timestamp TEXT,
            FOREIGN KEY(para_id) REFERENCES paragraphs(id)
        )""")
        # Mots
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id TEXT PRIMARY KEY,
            word TEXT,
            frequency INTEGER DEFAULT 1,
            doc_id TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")
        # Syllabes
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllables (
            id TEXT PRIMARY KEY,
            syllable TEXT,
            word_id TEXT,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )""")
        # Caractères
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
            ref_id TEXT,
            ref_type TEXT,
            PRIMARY KEY (concept, ref_id)
        )""")
        self.conn.commit()

    # -------------------------------------------------
    # CHARGEMENT DEPUIS SQLite VERS self.memory
    # -------------------------------------------------
    def load_memory_from_db(self):
        cursor = self.conn.cursor()
        # Phrases
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
        # Paragraphes
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

    def _serialize_embedding(self, emb):
        return json.dumps(emb.tolist())

    def _deserialize_embedding(self, data):
        if data is None:
            return None
        return np.array(json.loads(data))

    # -------------------------------------------------
    # CONSTRUCTION DES INDEX
    # -------------------------------------------------
    def build_embedding_matrix(self):
        vectors = [m["embedding"] for m in self.memory if m["embedding"] is not None]
        if vectors:
            self.embeddings_matrix = np.vstack(vectors)
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            self.embeddings_matrix /= np.where(norms == 0, 1.0, norms)
        else:
            self.embeddings_matrix = None

    def extract_concepts(self, text):
        """Extrait des concepts (mots longs, hors stop words)."""
        words = re.findall(r"\b\w+\b", text.lower())
        stop = {"le","la","les","de","des","du","un","une","et","en","dans","est","pour","que","qui","par","sur","avec","ce","cet","cette","ces","je","tu","il","elle","nous","vous","ils","elles","au","aux","a","à","ou","où","donc","car","ni","mais","or","si","the","and","of","to","in","for","on","with","by","as","at","from"}
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
    def semantic_split(self, text):
        """Découpe un long texte en sections significatives."""
        sections = re.split(r"\n\s*\d+[.)]\s+|\n—|\n\n", text)
        return [s.strip() for s in sections if len(s.strip()) > 50]

    # -------------------------------------------------
    # APPRENTISSAGE
    # -------------------------------------------------
    def learn(self, text, source="text", doc_id=None):
        """Apprend un texte : segmentation, embedding, stockage."""
        blocks = self.semantic_split(text)
        if not blocks:
            blocks = [text[:1000]]  # fallback
        timestamp = str(datetime.datetime.now())
        
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO documents (id, name, timestamp, source) VALUES (?,?,?,?)",
                           (doc_id, source, timestamp, "direct"))
            self.conn.commit()

        for block in blocks:
            embedding = self.model.encode(block)
            mem_id = str(uuid.uuid4())
            
            # Cache mémoire
            self.memory.append({
                "id": mem_id,
                "timestamp": timestamp,
                "text": block,
                "embedding": embedding,
                "source": source,
                "doc_id": doc_id
            })
            
            # Paragraphe
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO paragraphs (id, doc_id, text, embedding, timestamp) VALUES (?,?,?,?,?)",
                (mem_id, doc_id, block, self._serialize_embedding(embedding), timestamp)
            )
            
            # Phrases
            sentences = split_sentences(block)
            for sent in sentences:
                sent_emb = self.model.encode(sent)
                sent_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO sentences (id, para_id, text, embedding, timestamp) VALUES (?,?,?,?,?)",
                    (sent_id, mem_id, sent, self._serialize_embedding(sent_emb), timestamp)
                )
                # Mots
                words = split_words(sent)
                for w in words:
                    w_id = hash_text(w)
                    cursor.execute(
                        "INSERT OR IGNORE INTO words (id, word, frequency, doc_id) VALUES (?,?,?,?)",
                        (w_id, w, 1, doc_id)
                    )
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
            
            # Concepts
            concepts = self.extract_concepts(block)
            for c in concepts:
                cursor.execute(
                    "INSERT OR IGNORE INTO concepts (concept, ref_id, ref_type) VALUES (?,?,?)",
                    (c, mem_id, "paragraph")
                )
            
            self.conn.commit()
        
        # Reconstruction des index
        self.build_embedding_matrix()
        self.build_concept_index()
        return len(blocks)

    # -------------------------------------------------
    # EXTRACTION TEXTE DEPUIS FICHIER
    # -------------------------------------------------
    def extract_text_from_file(self, file):
        """Extrait le texte d'un fichier uploadé (objet binaire)."""
        text = ""
        filename = file.name.lower()
        if filename.endswith('.txt') or file.type == 'text/plain':
            text = file.read().decode('utf-8')
        elif filename.endswith('.pdf') or 'pdf' in file.type:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        elif filename.endswith('.docx') or 'word' in file.type:
            doc = docx.Document(file)
            for p in doc.paragraphs:
                text += p.text + "\n"
        elif filename.endswith('.csv') or 'csv' in file.type:
            df = pd.read_csv(file)
            text = df.to_string()
        elif filename.endswith(('.xls', '.xlsx')) or 'excel' in file.type:
            df = pd.read_excel(file)
            text = df.to_string()
        else:
            try:
                text = file.read().decode('utf-8')
            except:
                text = ""
        return text

    def learn_document(self, uploaded_file):
        """Apprend à partir d'un fichier uploadé."""
        text = self.extract_text_from_file(uploaded_file)
        if not text.strip():
            return 0
        doc_id = str(uuid.uuid4())
        timestamp = str(datetime.datetime.now())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, name, timestamp, source) VALUES (?,?,?,?)",
            (doc_id, uploaded_file.name, timestamp, uploaded_file.type or 'unknown')
        )
        self.conn.commit()
        return self.learn(text, source=uploaded_file.name, doc_id=doc_id)

    # -------------------------------------------------
    # RECHERCHES
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
    # RAISONNEMENT
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
    # STATISTIQUES
    # -------------------------------------------------
    def stats(self):
        cursor = self.conn.cursor()
        sentences = cursor.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        paragraphs = cursor.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        documents = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        words = cursor.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        return {
            "sentences": sentences,
            "paragraphs": paragraphs,
            "documents": documents,
            "words": words
        }

# ==========================================
# APPLICATION STREAMLIT
# ==========================================
st.set_page_config(page_title="Oracle TTU-MC³", layout="wide")

# Initialisation du moteur en cache (une seule fois)
@st.cache_resource
def get_engine():
    return OracleEngine()

engine = get_engine()

# Sidebar avec quelques infos
st.sidebar.title("🧠 Oracle TTU-MC³")
st.sidebar.markdown("Mémoire sémantique triadique")
st.sidebar.markdown("---")
st.sidebar.write("### Statistiques")
stats = engine.stats()
st.sidebar.write(f"**Documents :** {stats['documents']}")
st.sidebar.write(f"**Paragraphes :** {stats['paragraphs']}")
st.sidebar.write(f"**Phrases :** {stats['sentences']}")
st.sidebar.write(f"**Mots :** {stats['words']}")

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["📚 Apprentissage", "🔍 Recherche", "📊 Détails"])

with tab1:
    st.header("Apprentissage")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Texte brut")
        texte = st.text_area("Entrez un texte à mémoriser", height=200)
        if st.button("Apprendre le texte"):
            if texte.strip():
                nb = engine.learn(texte, source="streamlit_text")
                st.success(f"{nb} bloc(s) appris !")
            else:
                st.warning("Veuillez entrer un texte.")
    
    with col2:
        st.subheader("Upload de fichier")
        uploaded_file = st.file_uploader("Choisissez un fichier", 
                                         type=['txt','pdf','docx','csv','xls','xlsx'])
        if uploaded_file is not None:
            if st.button("Apprendre le fichier"):
                nb = engine.learn_document(uploaded_file)
                st.success(f"{nb} bloc(s) appris depuis {uploaded_file.name}")

with tab2:
    st.header("Recherche contextuelle")
    question = st.text_input("Posez votre question")
    if st.button("Interroger l'oracle"):
        if question.strip():
            reponse = engine.reason(question)
            st.markdown("### Réponse")
            st.write(reponse)
        else:
            st.warning("Veuillez entrer une question.")

with tab3:
    st.header("Détails de la base")
    st.json(engine.stats())
    
    # Afficher quelques échantillons de la mémoire
    st.subheader("Échantillon de la mémoire (derniers paragraphes)")
    cursor = engine.conn.cursor()
    cursor.execute("SELECT text FROM paragraphs ORDER BY timestamp DESC LIMIT 5")
    rows = cursor.fetchall()
    for i, (text,) in enumerate(rows):
        with st.expander(f"Paragraphe {i+1}"):
            st.write(text[:500] + "..." if len(text) > 500 else text)