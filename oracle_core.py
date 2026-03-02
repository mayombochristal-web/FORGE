# =====================================================
# 🧠 ORACLE V6 — CŒUR COGNITIF UNIFIÉ
# Facultés : Perception, Attention, Workspace, Homéostasie,
#            Génération, Métacognition, Apprentissage, Sommeil
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
        self.lexicon = self._load_lex()
        self.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.5}
        self.dialog_memory = deque(maxlen=60)          # workspace conscient
        self.ghost_memory = {}                          # mémoire fantôme
        self.ghost_activity = 0.0
        self.green_state = 0.0                           # bruit homéostasique
        self.hippocampus = []                            # buffer d'apprentissage
        self.last_sleep = time.time()

    # ---------- Persistance ----------
    def _load_lex(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_lex(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.lexicon, f, indent=2, ensure_ascii=False)

    # ---------- Perception ----------
    def perceive(self, raw_input):
        """Nettoie et tokenise l'entrée."""
        if not raw_input:
            return []
        # Nettoyage basique
        text = raw_input.lower().strip()
        words = text.split()
        return words

    # ---------- Attention (Thalamus) ----------
    def attend(self):
        """Retourne le seed contextuel le plus pertinent."""
        context_words = list(self.dialog_memory)
        if not context_words:
            # Si workspace vide, prendre un mot au hasard dans le lexique
            if self.lexicon:
                return random.choice(list(self.lexicon.keys()))
            return ""
        # Compter les mots dans le contexte
        flat = " ".join(context_words).split()
        candidates = [w for w in flat if w in self.lexicon]
        if candidates:
            return Counter(candidates).most_common(1)[0][0]
        # Fallback : premier mot du dernier message
        last_msg = context_words[-1].split()
        return last_msg[0] if last_msg else ""

    # ---------- Workspace ----------
    def workspace_add(self, message):
        self.dialog_memory.append(message)

    # ---------- Homéostasie (régulation Φ + green noise) ----------
    def _green_noise(self):
        self.green_state = 0.92 * self.green_state + 0.08 * random.uniform(-1, 1)
        return abs(self.green_state) < 0.25   # porte de consolidation

    def regulate(self, excitation):
        """Met à jour les paramètres Φ en fonction de l'excitation."""
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation*0.15 - 0.01))
        self.phi["phi_c"] = min(1, max(0.1, self.phi["phi_c"] + excitation*0.3 - 0.03))
        self.phi["phi_d"] = min(1, max(0.1, self.phi["phi_d"] + 0.02 - excitation*0.05))
        # Normalisation
        total = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= total
        # Appliquer le green noise (sans effet direct sur phi, juste pour la consolidation)
        self._green_noise()

    # ---------- Génération corticale ----------
    def _ghost_retrieve(self, word):
        """Récupère une suggestion depuis la mémoire fantôme."""
        if word not in self.ghost_memory or not self.ghost_memory[word]:
            return None
        options = self.ghost_memory[word]
        if random.random() < self.phi["phi_d"]:
            return random.choices(list(options.keys()), weights=list(options.values()))[0]
        return max(options, key=options.get)

    def _associative_layer(self, word):
        """Choix du mot suivant depuis le lexique principal."""
        if word not in self.lexicon:
            return word
        opts = self.lexicon[word]
        if random.random() < self.phi["phi_c"]:
            return random.choices(list(opts.keys()), weights=list(opts.values()))[0]
        return max(opts, key=opts.get)

    def generate(self, seed):
        """Produit une réponse à partir d'un mot graine."""
        if not self.lexicon:
            return "Mémoire vide. Nourrissez-moi."
        words = [seed]
        used = set(words)
        length = int(8 + self.phi["phi_m"] * 35)

        for _ in range(length):
            current = words[-1]
            if current not in self.lexicon:
                break

            # Choix principal
            main_next = self._associative_layer(current)

            # Influence fantôme
            ghost_next = self._ghost_retrieve(current)
            if ghost_next and ghost_next != main_next:
                if random.random() < self.phi["phi_d"] * 0.5:
                    self.ghost_activity = 0.8 * self.ghost_activity + 0.2
                    nxt = ghost_next
                else:
                    nxt = main_next
            else:
                nxt = main_next

            self.ghost_activity *= 0.95

            # Évitement de boucle (rôle de phi_d)
            if nxt in used and random.random() < self.phi["phi_d"]:
                break

            words.append(nxt)
            used.add(nxt)

        return " ".join(words).capitalize() + "."

    # ---------- Métacognition ----------
    def meta_observe(self, response):
        """Analyse la réponse et ajuste phi si nécessaire."""
        words = response.split()
        if len(words) < 3:
            return
        # Détection de répétition excessive
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:   # trop de répétitions
            self.phi["phi_d"] = min(1, self.phi["phi_d"] * 1.1)
        # Longueur trop courte ?
        if len(words) < 5:
            self.phi["phi_m"] = min(1, self.phi["phi_m"] * 1.05)
        # Normalisation après ajustements
        total = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= total

    # ---------- Apprentissage (hippocampe) ----------
    def learn_user(self, text, importance=1.0):
        """Apprentissage à partir d'un message utilisateur."""
        words = text.lower().split()
        if len(words) < 2:
            return
        energy = math.sqrt(sum(v*v for v in self.phi.values())) * importance
        # Stockage temporaire dans l'hippocampe
        self.hippocampus.append((words, energy))
        # Apprentissage fantôme immédiat (faible)
        self._ghost_learn(text)

    def _ghost_learn(self, text):
        words = text.lower().split()
        if len(words) < 2:
            return
        energy = math.sqrt(sum(v*v for v in self.phi.values())) * 0.3
        for a, b in zip(words, words[1:]):
            self.ghost_memory.setdefault(a, {})
            self.ghost_memory[a][b] = self.ghost_memory[a].get(b, 0) + energy
            if self.ghost_memory[a][b] > 5:
                self.ghost_memory[a][b] = 5

    def learn_self(self, text, importance=0.3):
        """Auto‑apprentissage sur la réponse générée."""
        self.learn_user(text, importance)

    def consolidate(self):
        """Transfère l'hippocampe vers la mémoire à long terme."""
        if not self.hippocampus:
            return
        # Consolidation uniquement si la porte green noise est ouverte
        if not self._green_noise():
            return
        for words, energy in self.hippocampus:
            for a, b in zip(words, words[1:]):
                self.lexicon.setdefault(a, {})
                self.lexicon[a][b] = self.lexicon[a].get(b, 0) + energy
        self.hippocampus.clear()
        # Consolidation fantôme
        self._ghost_consolidate()
        self._save_lex()

    def _ghost_consolidate(self):
        new_ghost = {}
        for w, cons in self.ghost_memory.items():
            filtered = {t: v * 0.95 for t, v in cons.items() if v > 0.8}
            if filtered:
                new_ghost[w] = filtered
        self.ghost_memory = new_ghost

    # ---------- Sommeil ----------
    def sleep_cycle(self):
        """Nettoyage profond de la mémoire (oubli)."""
        threshold = 1.5
        ban = ["http","www","uni00a0",".pdf",".docx","____"]
        new_lex = {}
        for w, con in self.lexicon.items():
            if len(w) < 2 or len(w) > 30 or any(b in w for b in ban):
                continue
            new_con = {t:v for t,v in con.items()
                       if v >= threshold and not any(b in t for b in ban)}
            if new_con:
                new_lex[w] = new_con
        self.lexicon = new_lex
        self._save_lex()
        self.last_sleep = time.time()

    def auto_sleep(self):
        """Déclenche un sommeil si nécessaire (intervalle > 1h)."""
        if time.time() - self.last_sleep > 3600:  # 1 heure
            self.sleep_cycle()

    # ---------- Pipeline complet ----------
    def process_input(self, user_input, is_user=True):
        """
        Exécute le pipeline cognitif complet.
        Retourne la réponse générée.
        """
        # 1. Perception
        words = self.perceive(user_input)
        if not words:
            return ""

        # 2. Ajout au workspace (mémoire de travail)
        self.workspace_add(user_input)

        # 3. Attention : déterminer le seed contextuel
        seed = self.attend()

        # 4. Régulation homéostasique (excitation basée sur la longueur)
        excitation = min(1, len(user_input) / 200)
        self.regulate(excitation)

        # 5. Génération de la réponse
        response = self.generate(seed)

        # 6. Métacognition sur la réponse
        self.meta_observe(response)

        # 7. Apprentissage (utilisateur et auto)
        self.learn_user(user_input, importance=1.1 if is_user else 1.0)
        self.learn_self(response, importance=0.3)

        # 8. Consolidation (transfert hippocampe → lexique)
        self.consolidate()

        # 9. Ajout de la réponse au workspace
        self.workspace_add(response)

        # 10. Vérification sommeil automatique
        self.auto_sleep()

        return response