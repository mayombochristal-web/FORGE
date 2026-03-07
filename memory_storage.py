import os
import json
from datetime import datetime

BASE = "oracle_memory"
MEM = os.path.join(BASE, "memories")
COMP = os.path.join(BASE, "compressed")
INDEX_PATH = os.path.join(BASE, "index.json")

def init():
    os.makedirs(MEM, exist_ok=True)
    os.makedirs(COMP, exist_ok=True)
    if not os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "w", encoding="utf8") as f:
            json.dump({}, f)

def save(text, vector, source):
    t = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    data = {
        "text": text,
        "vector": vector,
        "source": source,
        "time": t
    }
    path = os.path.join(MEM, f"memory_{t}.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path

def load_vector(path):
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return data["vector"], data["text"]
    except:
        return None, None

def load_all_metadata():
    rows = []
    for fname in os.listdir(MEM):
        if fname.endswith(".json"):
            path = os.path.join(MEM, fname)
            try:
                with open(path, "r", encoding="utf8") as f:
                    data = json.load(f)
                rows.append(data)
            except:
                continue
    return rows

def count_memories():
    return len([f for f in os.listdir(MEM) if f.endswith(".json")])