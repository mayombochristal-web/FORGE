import os
import json
import uuid
import datetime
import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from github import Github

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

    # ------------------------------------------------

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

    # ------------------------------------------------

    def save_memory(self, data):

        uid = str(uuid.uuid4())

        filename = f"{MEMORY_PATH}/{uid}.json"

        content = json.dumps(data, indent=2)

        self.repo.create_file(
            filename,
            f"memory {uid}",
            content
        )

    # ------------------------------------------------

    def stats(self):

        return len(self.memory)

    # ------------------------------------------------
    # SEGMENTATION INTELLIGENTE
    # ------------------------------------------------

    def semantic_split(self, text):

        sections = re.split(r"\n\s*\d+\s*—", text)

        chunks = []

        for s in sections:

            s = s.strip()

            if len(s) > 200:

                chunks.append(s)

        return chunks

    # ------------------------------------------------

    def learn(self, text):

        blocks = self.semantic_split(text)

        for block in blocks:

            embedding = self.model.encode(block).tolist()

            data = {

                "id": str(uuid.uuid4()),
                "timestamp": str(datetime.datetime.now()),
                "text": block,
                "embedding": embedding,
                "type": "knowledge"

            }

            self.memory.append(data)

            self.save_memory(data)

        return len(blocks)

    # ------------------------------------------------

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

        return best

    # ------------------------------------------------

    def report(self):

        texts = [m["text"] for m in self.memory]

        words = " ".join(texts).split()

        concepts = Counter(words).most_common(10)

        return {

            "souvenirs_totaux": len(self.memory),

            "concepts": concepts
        }
