import os
import json
import uuid
import datetime
import numpy as np
import re
import pandas as pd
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
from github import Github

import PyPDF2
import docx

MEMORY_PATH = "oracle_memory"


class OracleEngine:

    def __init__(self):

        # ===============================
        # MODEL EMBEDDINGS
        # ===============================

        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # ===============================
        # GITHUB STORAGE
        # ===============================

        try:
            self.github = Github(os.getenv("GITHUB_TOKEN"))
            self.repo = self.github.get_repo(os.getenv("GITHUB_REPO"))
        except:
            self.repo = None

        self.memory = []
        self.embeddings_matrix = None

        self.concept_index = defaultdict(list)

        self.load_memory()
        self.build_embedding_matrix()
        self.build_concept_index()

    # =====================================================
    # LOAD MEMORY
    # =====================================================

    def load_memory(self):

        if self.repo is None:
            return

        try:

            files = self.repo.get_contents(MEMORY_PATH)

            for f in files:

                content = self.repo.get_contents(f.path)

                data = json.loads(content.decoded_content)

                if "source" not in data:
                    data["source"] = "legacy"

                self.memory.append(data)

        except:
            pass

    # =====================================================
    # BUILD EMBEDDING MATRIX (V18 OPTIMIZATION)
    # =====================================================

    def build_embedding_matrix(self):

        vectors = []

        for m in self.memory:

            if "embedding" in m:
                vectors.append(m["embedding"])

        if vectors:

            self.embeddings_matrix = np.array(vectors)

            norms = np.linalg.norm(self.embeddings_matrix, axis=1)

            self.embeddings_matrix = (
                self.embeddings_matrix / norms[:, None]
            )

    # =====================================================
    # SAVE MEMORY
    # =====================================================

    def save_memory(self, data):

        if self.repo is None:
            return

        uid = data["id"]

        filename = f"{MEMORY_PATH}/{uid}.json"

        content = json.dumps(data, indent=2, ensure_ascii=False)

        try:

            self.repo.create_file(
                filename,
                f"oracle memory {uid}",
                content
            )

        except:
            pass

    # =====================================================
    # CONCEPT EXTRACTION
    # =====================================================

    def extract_concepts(self, text):

        words = re.findall(r"\b\w+\b", text.lower())

        stop = {
            "le","la","les","de","des","du",
            "un","une","et","en","dans",
            "est","pour","que"
        }

        concepts = [w for w in words if w not in stop and len(w) > 4]

        return list(set(concepts))

    # =====================================================
    # BUILD CONCEPT INDEX
    # =====================================================

    def build_concept_index(self):

        self.concept_index = defaultdict(list)

        for m in self.memory:

            concepts = self.extract_concepts(m["text"])

            for c in concepts:
                self.concept_index[c].append(m)

    # =====================================================
    # TEXT SEGMENTATION
    # =====================================================

    def semantic_split(self, text):

        sections = re.split(r"\n\s*\d+\s*—|\n\n", text)

        chunks = []

        for s in sections:

            s = s.strip()

            if len(s) > 200:
                chunks.append(s)

        return chunks

    # =====================================================
    # LEARN TEXT
    # =====================================================

    def learn(self, text, source="text"):

        blocks = self.semantic_split(text)

        for block in blocks:

            embedding = self.model.encode(block)

            data = {

                "id": str(uuid.uuid4()),
                "timestamp": str(datetime.datetime.now()),
                "text": block,
                "embedding": embedding.tolist(),
                "source": source

            }

            self.memory.append(data)

            self.save_memory(data)

        self.build_embedding_matrix()
        self.build_concept_index()

        return len(blocks)

    # =====================================================
    # DOCUMENT EXTRACTION
    # =====================================================

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

    # =====================================================
    # LEARN DOCUMENT
    # =====================================================

    def learn_document(self, file):

        text = self.extract_text(file)

        return self.learn(text, source=file.name)

    # =====================================================
    # VECTOR SEARCH (FAST V18)
    # =====================================================

    def vector_search(self, question, top_k=5):

        if self.embeddings_matrix is None:
            return []

        q_embed = self.model.encode(question)

        q_embed = q_embed / np.linalg.norm(q_embed)

        scores = np.dot(self.embeddings_matrix, q_embed)

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_idx:

            results.append(self.memory[idx])

        return results

    # =====================================================
    # CONCEPT SEARCH
    # =====================================================

    def concept_search(self, question):

        concepts = self.extract_concepts(question)

        results = []

        for c in concepts:

            if c in self.concept_index:

                results.extend(self.concept_index[c])

        return results

    # =====================================================
    # REASONING
    # =====================================================

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

    # =====================================================
    # STATS
    # =====================================================

    def stats(self):

        sources = Counter(
            [m.get("source","unknown") for m in self.memory]
        )

        return {
            "souvenirs": len(self.memory),
            "sources": dict(sources)
        }
