import re
import math
from collections import Counter
from oracle_memory_manager import MemoryManager
from oracle_attention_engine import AttentionEngine
from reasoning_engine import ReasoningEngine
from concept_graph import ConceptGraph
from analytics_engine import AnalyticsEngine
from ttu_file_scanner import scan_file

STOPWORDS = {
    "les","des","une","dans","pour","avec",
    "qui","que","est","sur","pas","plus",
    "par","comme","mais","donc","car"
}

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zàâéèêëîïôûùüç]{3,}", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def vectorize(tokens):
    v = {}
    for t in tokens:
        v[t] = v.get(t, 0) + 1
    return v

def cosine(v1, v2):
    inter = set(v1) & set(v2)
    num = sum(v1[x] * v2[x] for x in inter)
    s1 = sum(v**2 for v in v1.values())
    s2 = sum(v**2 for v in v2.values())
    denom = math.sqrt(s1) * math.sqrt(s2)
    if denom == 0:
        return 0
    return num / denom

class OracleEngine:
    def __init__(self):
        self.memory = MemoryManager()
        self.attention = AttentionEngine()
        self.reasoner = ReasoningEngine()
        self.graph = ConceptGraph()
        self.analytics = AnalyticsEngine()

    def learn(self, text):
        tokens = tokenize(text)
        score = self.attention.score(tokens)
        if score > 3:
            vector = vectorize(tokens)
            self.memory.store(text, vector, "user")
            self.graph.update(tokens)

    def learn_document(self, file):
        text = scan_file(file)
        words = text.split()
        chunks = []
        for i in range(0, len(words), 120):
            chunk = " ".join(words[i:i+120])
            if len(chunk) > 80:
                chunks.append(chunk)
        learned = 0
        for c in chunks:
            tokens = tokenize(c)
            if len(tokens) > 5:
                vector = vectorize(tokens)
                self.memory.store(c, vector, "document")
                self.graph.update(tokens)
                learned += 1
        return learned

    def reason(self, question):
        tokens = tokenize(question)
        qvec = vectorize(tokens)
        results = self.memory.search(qvec)
        reasoning = self.reasoner.build_reasoning(question, results)
        return reasoning

    def stats(self):
        return self.memory.count()

    def report(self):
        rows = self.memory.all()
        sources = Counter()
        concepts = Counter()
        for r in rows:
            sources[r["source"]] += 1
            tokens = tokenize(r["text"])
            for t in tokens:
                concepts[t] += 1
        return {
            "souvenirs_totaux": len(rows),
            "sources": dict(sources),
            "concepts": dict(concepts.most_common(20))
        }