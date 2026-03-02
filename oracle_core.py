# =====================================================
# 🧠 ORACLE CORE — V3.2 Ω STABLE BRAIN
# =====================================================

import json
import os
import random
import math
from collections import deque
from datetime import datetime

# =====================================================
# MEMORY PATHS
# =====================================================

MEMORY_FILE = "memory.json"

# =====================================================
# UTILITIES
# =====================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem[-500:], f, indent=2, ensure_ascii=False)

# =====================================================
# ORACLE BRAIN
# =====================================================

class OracleBrain:

    def __init__(self):
        self.long_memory = load_memory()
        self.dialog_memory = deque(maxlen=40)
        self.phi = 0.5
        self.energy = 0.5

    # ----------------------------
    # THALAMUS (V4.5 Ω)
    # ----------------------------
    def contextual_seed(self):

        text = " ".join(self.dialog_memory[-6:])

        words = text.split()
        if not words:
            return ""

        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        return max(freq, key=freq.get)

    # ----------------------------
    # GREEN NOISE (V3.1)
    # ----------------------------
    def green_noise(self):
        return random.uniform(-0.05, 0.05)

    # ----------------------------
    # HOMEOSTASIS (V6 Ω)
    # ----------------------------
    def regulate(self):
        self.phi += self.green_noise()
        self.phi = max(0.1, min(1.0, self.phi))

    # ----------------------------
    # LEARNING SYSTEM
    # ----------------------------
    def learn(self, text, weight=1.0):

        self.long_memory.append({
            "text": text,
            "weight": weight,
            "time": str(datetime.now())
        })

        save_memory(self.long_memory)

    # ----------------------------
    # GENERATION ENGINE
    # ----------------------------
    def generate(self, user_input):

        seed = self.contextual_seed()

        base = f"Seed:{seed} | Φ={round(self.phi,2)}"

        thought_length = int(8 + self.phi * 35)

        words = user_input.split()

        generated = []

        for i in range(thought_length):
            if words:
                generated.append(random.choice(words))
            else:
                generated.append(seed)

        reply = base + "\n" + " ".join(generated)

        return reply

    # ----------------------------
    # MAIN PIPELINE
    # ----------------------------
    def process(self, user_input):

        self.dialog_memory.append(user_input)

        self.regulate()

        reply = self.generate(user_input)

        self.dialog_memory.append(reply)

        # dual learning V4.5 Ω
        self.learn(user_input, 1.0)
        self.learn(reply, 0.3)

        return reply


# =====================================================
# GLOBAL BRAIN INSTANCE (CRITICAL)
# =====================================================

brain = OracleBrain()

# =====================================================
# EXPORTED API (DO NOT RENAME)
# =====================================================

def process_input(text: str):
    return brain.process(text)