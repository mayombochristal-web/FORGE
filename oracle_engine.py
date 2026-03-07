import os
import json
import uuid
import datetime
import numpy as np
import re
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from github import Github
import PyPDF2
import docx

MEMORY_PATH = "oracle_memory"

class OracleEngine:

    def __init__(self):

        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.github = Github(os.getenv("GITHUB_TOKEN"))
        self.repo = self.github.get_repo(os.getenv("GITHUB_REPO"))

        self.memory = []

        self.load_memory()

    # =====================================================
    # CHARGEMENT MEMOIRE GITHUB
    # =====================================================

    def load_memory(self):

        try:

            files = self.repo.get_contents(MEMORY_PATH)

            for f in files:

                data = json.loads(
                    self.repo.get_contents(f.path).decoded_content
                )

                self.memory.append(data)

        except:
            pass

    # =====================================================
    # SAUVEGARDE MEMOIRE
    # =====================================================

    def save_memory(self, data):

        uid = str(uuid.uuid4())

        filename = f"{MEMORY_PATH}/{uid}.json"

        content = json.dumps(data, indent=2, ensure_ascii=False)

        self.repo.create_file(
            filename,
            f"oracle memory {uid}",
            content
        )

    # =====================================================
    # STATS
    # =====================================================

    def stats(self):

        return len(self.memory)

    # =====================================================
    # SEGMENTATION SEMANTIQUE
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
    # APPRENTISSAGE TEXTE
    # =====================================================

    def learn(self, text, source="text"):

        blocks = self.semantic_split(text)

        for block in blocks:

            embedding = self.model.encode(block).tolist()

            data = {

                "id": str(uuid.uuid4()),
                "timestamp": str(datetime.datetime.now()),
                "text": block,
                "embedding": embedding,
                "source": source

            }

            self.memory.append(data)

            self.save_memory(data)

        return len(blocks)

    # =====================================================
    # EXTRACTION DOCUMENT
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

        return text

    # =====================================================
    # APPRENTISSAGE DOCUMENT
    # =====================================================

    def learn_document(self, file):

        text = self.extract_text(file)

        blocks = self.learn(text, source=file.name)

        return blocks

    # =====================================================
    # RAISONNEMENT
    # =====================================================

    def reason(self, question):

        q_embed = self.model.encode(question)

        best = None
        score_max = -1

        for m in self.memory:

            emb = np.array(m["embedding"])

            score = np.dot(q_embed, emb) / (
                np.linalg.norm(q_embed) *
                np.linalg.norm(emb)
            )

            if score > score_max:

                score_max = score
                best = m["text"]

        if best:

            return best

        return "Aucune connaissance pertinente trouvée."

    # =====================================================
    # RAPPORT MEMOIRE
    # =====================================================

    def report(self):

        texts = [m["text"] for m in self.memory]

        words = " ".join(texts).split()

        concepts = Counter(words).most_common(10)

        sources = Counter([m["source"] for m in self.memory])

        return {

            "souvenirs_totaux": len(self.memory),

            "sources": dict(sources),

            "concepts": concepts

        }
