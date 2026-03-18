#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle Memory Engine TTU-MC³
Fusion complète : Flask + OracleEngine (version unifiée)

Auteur : Mayombo Idiedie Christ Aldo & Scott Brooz
Date : 21 février 2026

Fonctionnalités :
- Apprentissage de textes bruts, fichiers (PDF, DOCX, CSV, Excel, TXT)
- Segmentation sémantique, embeddings multilingues
- Stockage SQLite avec tables fines (paragraphes, phrases, mots, syllabes, caractères, concepts)
- Recherche vectorielle + recherche par concepts
- Réponse contextuelle (top 3 passages)
- Statistiques et sauvegarde automatique sur GitHub (optionnel)
"""

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
import requests
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_FOLDER = "oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle.db")
GITHUB_REPO = os.getenv("ORACLE_GITHUB_REPO")      # optionnel
GITHUB_TOKEN = os.getenv("ORACLE_GITHUB_TOKEN")    # optionnel
SATURATION_THRESHOLD = 0.45  # pour le diagramme de phase (non utilisé ici, mais gardé)
DT = 0.01                     # pas de temps (réserve)

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

# ==========================================
# UTILITAIRES
# ==========================================
def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def split_syllables(word):
    """Découpe un mot en syllabes (approximatif pour le français/anglais)."""
    return re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]|$)', word.lower())

def split_sentences(text):
    """Découpe un texte en phrases."""
    return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

def split_paragraphs(text):
    """Découpe un texte en paragraphes (séparés par double saut de ligne)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_words(text):
    """Extrait les mots d'un texte (minuscules)."""
    return re.findall(r'\b\w+\b', text.lower())

def push_db_to_github():
    """Pousse le fichier SQLite vers GitHub (si configuré)."""
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return
    with open(DB_PATH, "rb") as f:
        content = f.read()
    b64 = base64.b64encode(content).decode()
    filename = os.path.basename(DB_PATH)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"message": "oracle memory update", "content": b64}
    try:
        requests.put(url, json=data, headers=headers)
    except Exception as e:
        print(f"GitHub push failed: {e}")

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
        
        # Cache mémoire (pour les recherches)
        self.memory = []          # liste de dicts {id, timestamp, text, embedding, source, parent...}
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
            embedding TEXT,      # stocké comme JSON
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
        # Mots (avec fréquence)
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
        # On charge les phrases
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
        # On charge les paragraphes
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
        """Convertit un numpy array en JSON string."""
        return json.dumps(emb.tolist())

    def _deserialize_embedding(self, data):
        """Reconstruit un numpy array depuis JSON."""
        if data is None:
            return None
        return np.array(json.loads(data))

    # -------------------------------------------------
    # CONSTRUCTION DES INDEX
    # -------------------------------------------------
    def build_embedding_matrix(self):
        vectors = [m["embedding"] for m in self.memory if "embedding" in m and m["embedding"] is not None]
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
        """Découpe un long texte en sections significatives (basé sur titres ou doubles sauts)."""
        # Si le texte contient des titres comme "1. Introduction" ou "—", on coupe.
        sections = re.split(r"\n\s*\d+[.)]\s+|\n—|\n\n", text)
        return [s.strip() for s in sections if len(s.strip()) > 50]

    # -------------------------------------------------
    # APPRENTISSAGE
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
                "embedding": embedding,
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
            
            # Découpage en phrases pour stockage fin
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
    # EXTRACTION TEXTE DEPUIS FICHIER
    # -------------------------------------------------
    def extract_text_from_file(self, file):
        """Extrait le texte d'un fichier uploadé."""
        text = ""
        filename = file.name.lower()
        if filename.endswith('.txt') or file.content_type == 'text/plain':
            text = file.read().decode('utf-8')
        elif filename.endswith('.pdf') or 'pdf' in file.content_type:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        elif filename.endswith('.docx') or 'word' in file.content_type:
            doc = docx.Document(file)
            for p in doc.paragraphs:
                text += p.text + "\n"
        elif filename.endswith('.csv') or 'csv' in file.content_type:
            df = pd.read_csv(file)
            text = df.to_string()
        elif filename.endswith(('.xls', '.xlsx')) or 'excel' in file.content_type:
            df = pd.read_excel(file)
            text = df.to_string()
        else:
            # fallback : tenter de lire comme texte
            try:
                text = file.read().decode('utf-8')
            except:
                text = ""
        return text

    def learn_document(self, file):
        """Apprend à partir d'un objet fichier (upload)."""
        text = self.extract_text_from_file(file)
        if not text.strip():
            return 0
        # Création d'une entrée document dans SQLite
        doc_id = str(uuid.uuid4())
        timestamp = str(datetime.datetime.now())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, name, timestamp, source) VALUES (?,?,?,?)",
            (doc_id, file.name, timestamp, file.content_type or 'unknown')
        )
        self.conn.commit()
        return self.learn(text, source=file.name, doc_id=doc_id)

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
            return "Aucune connaissance pertinente trouvée dans la base."
        return "\n\n".join(texts[:3])

    # -------------------------------------------------
    # STATISTIQUES
    # -------------------------------------------------
    def stats(self):
        cursor = self.conn.cursor()
        souvenirs = cursor.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        nb_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        nb_paragraphs = cursor.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        sources = Counter([m.get("source", "unknown") for m in self.memory])
        return {
            "sentences": souvenirs,
            "paragraphs": nb_paragraphs,
            "documents": nb_docs,
            "sources": dict(sources)
        }

    # -------------------------------------------------
    # NETTOYAGE
    # -------------------------------------------------
    def close(self):
        self.conn.close()

# ==========================================
# APPLICATION FLASK
# ==========================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Instance unique du moteur
engine = OracleEngine()

INDEX_HTML = '''
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Oracle TTU-MC³</title>
    <style>
        body { background: #0e1117; color: #00ff41; font-family: 'Courier New', monospace; padding: 20px; }
        .card { border: 1px solid #00ff41; padding: 15px; margin-bottom: 20px; background: rgba(0,255,65,0.05); }
        input, textarea, button { background: #000; color: #00ff41; border: 1px solid #00ff41; padding: 8px; margin: 5px 0; width: 100%; }
        button { background: #00ff41; color: #000; font-weight: bold; cursor: pointer; }
        hr { border-color: #00ff41; }
    </style>
</head>
<body>
    <h1>🌌 ORACLE MEMORY ENGINE [TTU-MC³]</h1>
    <div class="card">
        <h3>Statistiques</h3>
        <p><a href="/stats" style="color:#00ff41;">Voir les stats (JSON)</a></p>
    </div>
    <div class="card">
        <h3>Apprendre un texte</h3>
        <form action="/learn" method="post">
            <textarea name="text" rows="5" placeholder="Collez votre texte ici..."></textarea>
            <button type="submit">Apprendre</button>
        </form>
    </div>
    <div class="card">
        <h3>Uploader un fichier</h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Uploader</button>
        </form>
    </div>
    <div class="card">
        <h3>Poser une question</h3>
        <form action="/query" method="post">
            <input type="text" name="question" placeholder="Votre question...">
            <button type="submit">Questionner</button>
        </form>
    </div>
    <hr>
    <p><a href="/" style="color:#00ff41;">↻ Accueil</a></p>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(engine.stats())

@app.route('/learn', methods=['POST'])
def learn():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'Aucun texte fourni'}), 400
    nb_blocks = engine.learn(text, source='web_form')
    return jsonify({'message': f'{nb_blocks} bloc(s) appris avec succès'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    # Sauvegarder temporairement pour garantir la compatibilité (certaines bibliothèques ont besoin d'un chemin)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # On rouvre le fichier en mode binaire
        with open(tmp_path, 'rb') as f:
            class FakeFile:
                def __init__(self, fileobj, filename, content_type):
                    self.fileobj = fileobj
                    self.name = filename
                    self.content_type = content_type
                def read(self, *args, **kwargs):
                    return self.fileobj.read(*args, **kwargs)
            content_type = file.content_type or 'application/octet-stream'
            fake_file = FakeFile(f, file.filename, content_type)
            nb_blocks = engine.learn_document(fake_file)
    finally:
        os.unlink(tmp_path)

    return jsonify({'message': f'{nb_blocks} bloc(s) appris depuis le fichier {file.filename}'})

@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '')
    if not question:
        return jsonify({'error': 'Aucune question fournie'}), 400
    answer = engine.reason(question)
    return jsonify({'question': question, 'answer': answer})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Ferme proprement la connexion à la base (utile pour certains hébergeurs)."""
    engine.close()
    return jsonify({'message': 'Base fermée'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)