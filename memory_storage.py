import os
import json
from datetime import datetime

BASE_DIR = "oracle_memory"
MEMORY_DIR = os.path.join(BASE_DIR, "memories")

def init_storage():

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)


def save_memory(content, score):

    init_storage()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    memory = {
        "timestamp": timestamp,
        "score": score,
        "content": content
    }

    filename = f"memory_{timestamp}.json"

    path = os.path.join(MEMORY_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

    return path