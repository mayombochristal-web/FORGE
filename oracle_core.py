# =====================================================
# 🧠 ORACLE V6.1 — CŒUR COGNITIF UNIFIÉ & RELATIONNEL
# Facultés : Perception, Attention, Nexus Multi-fichiers,
#            Homéostasie, Génération, Apprentissage
# =====================================================

import random
import json
import os
import math
import time
from collections import deque, Counter

class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.lexicon = self._load_lex(self.memory_file)
        self.nexus_lexicon = {}                         # Lexique de référence croisée (Lecture seule)
        self.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}
        self.dialog_memory = deque(maxlen=60)          
        self.ghost_memory = {}                          
        self.ghost_activity = 0.0
        self.green_state = 0.0                           
        self.hippocampus = []                            
        self.last_sleep = time.time()

    # ---------- Persistance & Nexus ----------
    def _load_lex(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_lex(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.lexicon, f, indent=2, ensure_ascii=False)

    def cross_reference(self, autre_fichier):
        """
        Ouvre un second fichier en lecture seule pour enrichir la réflexion.
        Permet de créer des relations entre deux banques de données.
        """
        if autre_fichier and autre_fichier != self.memory_file:
            self.nexus_lexicon = self._load_lex(autre_fichier)
            return True
        self.nexus_lexicon = {}
        return False

    # ---------- Perception & Attention ----------
    def perceive(self, raw_input):
        if not raw_input: return []
        return raw_input.lower().strip().split()

    def attend(self):
        context_words = list(self.dialog_memory)
        if not context_words:
            return random.choice(list(self.lexicon.keys())) if self.lexicon else ""
        
        flat = " ".join(context_words).split()
        # L'attention vérifie aussi si le mot existe dans le Nexus
        candidates = [w for w in flat if w in self.lexicon or w in self.nexus_lexicon]
        if candidates:
            return Counter(candidates).most_common(1)[0][0]
        return context_words[-1].split()[0] if context_words else ""

    def workspace_add(self, message):
        self.dialog_memory.append(message)

    # ---------- Homéostasie ----------
    def _green_noise(self):
        self.green_state = 0.92 * self.green_state + 0.08 * random.uniform(-1, 1)
        return abs(self.green_state) < 0.25

    def regulate(self, excitation):
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation*0.15 - 0.01))
        self.phi["phi_c"] = min(1, max(0.1, self.phi["phi_c"] + excitation*0.3 - 0.03))
        self.phi["phi_d"] = min(1, max(0.1, self.phi["phi_d"] + 0.02 - excitation*0.05))
        total = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= total
        self._green_noise()

    # ---------- Génération Corticale & Relationnelle ----------
    def _nexus_layer(self, word):
        """Recherche une association dans le fichier de référence croisée."""
        if word not in self.nexus_lexicon:
            return None
        opts = self.nexus_lexicon[word]
        # On privilégie le Nexus si phi_c (créativité/cohésion) est élevé
        if random.random() < self.phi["phi_c"]:
            return random.choices(list(opts.keys()), weights=list(opts.values()))[0]
        return max(opts, key=opts.get)

    def _associative_layer(self, word):
        if word not in self.lexicon: return word
        opts = self.lexicon[word]
        if random.random() < self.phi["phi_c"]:
            return random.choices(list(opts.keys()), weights=list(opts.values()))[0]
        return max(opts, key=opts.get)

    def generate(self, seed):
        if not self.lexicon and not self.nexus_lexicon:
            return "Mémoire vide."
        
        words = [seed] if seed else ["oracle"]
        used = set(words)
        length = int(8 + self.phi["phi_m"] * 35)

        for _ in range(length):
            current = words[-1]
            
            # 1. Tentative de saut relationnel vers le Nexus (autre fichier)
            nexus_next = self._nexus_layer(current)
            main_next = self._associative_layer(current) if current in self.lexicon else None

            # 2. Arbitrage sémantique (Fusion des mémoires)
            if nexus_next and (not main_next or random.random() < 0.4):
                nxt = nexus_next
                self.ghost_activity = 0.7 * self.ghost_activity + 0.3 # Marqueur d'influence externe
            elif main_next:
                nxt = main_next
            else:
                break

            # Évitement de boucle
            if nxt in used and random.random() < self.phi["phi_d"]: break
            
            words.append(nxt)
            used.add(nxt)

        return " ".join(words).capitalize() + "."

    # ---------- Apprentissage ----------
    def learn_user(self, text, importance=1.0):
        words = text.lower().split()
        if len(words) < 2: return
        energy = math.sqrt(sum(v*v for v in self.phi.values())) * importance
        self.hippocampus.append((words, energy))

    def consolidate(self):
        if not self.hippocampus or not self._green_noise(): return
        for words, energy in self.hippocampus:
            for a, b in zip(words, words[1:]):
                self.lexicon.setdefault(a, {})
                self.lexicon[a][b] = self.lexicon[a].get(b, 0) + energy
        self.hippocampus.clear()
        self._save_lex()

    # ---------- Sommeil & Nettoyage ----------
    def sleep_cycle(self):
        self.lexicon = {w: {t: v for t, v in c.items() if v >= 1.5} 
                        for w, c in self.lexicon.items() if len(w) > 1}
        self._save_lex()
        self.last_sleep = time.time()

    # ---------- Pipeline Complet ----------
    def process_input(self, user_input, is_user=True):
        words = self.perceive(user_input)
        if not words: return ""

        self.workspace_add(user_input)
        seed = self.attend()
        self.regulate(min(1, len(user_input) / 200))
        
        response = self.generate(seed)
        
        self.learn_user(user_input, importance=1.1)
        self.consolidate()
        self.workspace_add(response)
        
        # Reset du Nexus après génération pour rester focalisé
        self.nexus_lexicon = {} 
        
        return response
