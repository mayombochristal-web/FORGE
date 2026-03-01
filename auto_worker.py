# =====================================================
# 🧠 ORACLE S+17 — SLEEP WORKER (∆k/k ENGINE)
# =====================================================

import pandas as pd
import numpy as np
import json, os, re, datetime, feedparser
from collections import Counter

MEM = "oracle_memory"

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

SOURCES = [
    "https://arxiv.org/rss/math.DS",
    "https://arxiv.org/rss/cs.AI",
    "https://feeds.feedburner.com/sciencedaily/computers_math/mathematics"
]

# -----------------------------------------------------
# LOAD MEMORY
# -----------------------------------------------------
def load_memory():
    if not os.path.exists(FILES["cortex"]):
        return None, None, None

    frag = pd.read_csv(FILES["fragments"])
    rel = json.load(open(FILES["relations"], encoding="utf-8"))
    ctx = json.load(open(FILES["cortex"], encoding="utf-8"))
    return frag, rel, ctx

# -----------------------------------------------------
# TOKENIZER
# -----------------------------------------------------
def clean_text(t):
    return re.sub(r"[^\wàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean_text(t).split() if len(w) > 1]

# -----------------------------------------------------
# ∆k/k REGULATOR
# -----------------------------------------------------
def delta_k_control(new_words, ctx):

    total_knowledge = max(ctx.get("age", 1), 1)
    delta_k = len(new_words)

    ratio = delta_k / total_knowledge

    ctx["delta_k"] = delta_k
    ctx["delta_ratio"] = ratio

    # seuil adaptatif TTU
    if ratio > 0.25:
        return False   # surcharge cognitive

    return True

# -----------------------------------------------------
# LEARNING
# -----------------------------------------------------
def worker_learn(text, frag, rel, ctx):

    words = tokenize(text)
    if not words:
        return frag, rel, ctx

    if not delta_k_control(words, ctx):
        print("⚠️ Apprentissage refusé (∆k/k trop élevé)")
        return frag, rel, ctx

    counts = Counter(words)

    memory = dict(zip(frag["fragment"], frag["count"]))

    for w, c in counts.items():
        memory[w] = memory.get(w, 0) + c

    frag = pd.DataFrame(memory.items(), columns=["fragment","count"])

    # relations
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        rel.setdefault(a,{})
        rel[a][b] = rel[a].get(b,0)+1

    # cortex update
    ctx["age"] += len(words)
    ctx["VS"] = 10 + np.log1p(ctx["age"])

    ctx.setdefault("timeline", []).append({
        "t": str(datetime.datetime.now()),
        "dk": ctx["delta_ratio"]
    })

    ctx["timeline"] = ctx["timeline"][-5000:]

    return frag, rel, ctx

# -----------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------
def run_worker():

    print("🧠 Sleep Cycle Started")

    frag, rel, ctx = load_memory()

    if frag is None:
        print("Mémoire absente.")
        return

    total = 0

    for url in SOURCES:

        feed = feedparser.parse(url)

        for entry in feed.entries[:3]:

            text = f"{entry.title} {entry.description}"

            frag, rel, ctx = worker_learn(text, frag, rel, ctx)

            total += len(tokenize(text))

    frag.to_csv(FILES["fragments"], index=False)
    json.dump(rel, open(FILES["relations"],"w",encoding="utf-8"))
    json.dump(ctx, open(FILES["cortex"],"w",encoding="utf-8"))

    print(f"✅ Sleep complete : {total} mots assimilés")

if __name__ == "__main__":
    run_worker()