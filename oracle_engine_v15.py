import os
import json
import uuid
import datetime
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from github import Github
import PyPDF2
import docx
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

MEMORY_PATH = "oracle_memory"

# ============================================================
# ORACLE ENGINE
# ============================================================

class OracleEngine:

    def __init__(self):

        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.github = Github(GITHUB_TOKEN)
        self.repo = self.github.get_repo(GITHUB_REPO)

        self.memory = []

        self.load_memory()

    # ========================================================
    # LOAD MEMORY
    # ========================================================

    def load_memory(self):

        try:

            contents = self.repo.get_contents(MEMORY_PATH)

            for file in contents:

                if file.name.endswith(".json"):

                    data = json.loads(
                        self.repo.get_contents(file.path).decoded_content
                    )

                    self.memory.append(data)

        except:
            pass

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    def save_memory(self, data):

        uid = str(uuid.uuid4())

        filename = f"{MEMORY_PATH}/{uid}.json"

        content = json.dumps(data, indent=2)

        try:

            self.repo.create_file(
                filename,
                f"memory {uid}",
                content
            )

        except:

            pass

    # ========================================================
    # STATS
    # ========================================================

    def stats(self):

        return len(self.memory)

    # ========================================================
    # LEARN TEXT
    # ========================================================

    def learn(self, text):

        embedding = self.model.encode(text).tolist()

        data = {
            "id": str(uuid.uuid4()),
            "type": "text",
            "timestamp": str(datetime.datetime.now()),
            "text": text,
            "embedding": embedding
        }

        self.memory.append(data)

        self.save_memory(data)

    # ========================================================
    # DOCUMENT INGESTION
    # ========================================================

    def learn_document(self, file):

        text = ""

        if file.type == "text/plain":

            text = file.read().decode()

        elif file.type == "application/pdf":

            pdf = PyPDF2.PdfReader(file)

            for page in pdf.pages:
                text += page.extract_text()

        elif "word" in file.type:

            doc = docx.Document(file)

            for p in doc.paragraphs:
                text += p.text + "\n"

        elif "csv" in file.type:

            df = pd.read_csv(file)

            text = df.to_string()

        blocks = self.chunk_text(text)

        for block in blocks:
            self.learn(block)

        return len(blocks)

    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    def chunk_text(self, text, size=500):

        chunks = []

        for i in range(0, len(text), size):

            chunks.append(text[i:i+size])

        return chunks

    # ========================================================
    # REASONING
    # ========================================================

    def reason(self, question):

        q_embed = self.model.encode(question)

        best_score = -1
        best_text = ""

        for mem in self.memory:

            emb = np.array(mem["embedding"])

            score = np.dot(q_embed, emb) / (
                np.linalg.norm(q_embed) *
                np.linalg.norm(emb)
            )

            if score > best_score:

                best_score = score
                best_text = mem["text"]

        return f"🧠 Mémoire associée :\n\n{best_text}"

    # ========================================================
    # REPORT
    # ========================================================

    def report(self):

        texts = [m["text"] for m in self.memory]

        words = " ".join(texts).split()

        concepts = Counter(words).most_common(10)

        sources = Counter([m["type"] for m in self.memory])

        return {

            "souvenirs_totaux": len(self.memory),

            "sources": dict(sources),

            "concepts": concepts
        }